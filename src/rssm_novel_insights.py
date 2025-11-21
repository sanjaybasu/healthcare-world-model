"""
RSSM Novel Insights Generation
Generates figures for:
1. Capacity-Demand Coupling
2. Shock Propagation
3. Intervention Cascades
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rssm_architecture import HealthcareRSSM
from rssm_architecture import HealthcareRSSM
from torch.utils.data import Dataset

class MedicaidDataset(Dataset):
    def __init__(self, data, sequence_length=2):
        self.data = data
        self.sequence_length = sequence_length
        self.person_ids = data['person_id'].unique()
        self.grouped = data.groupby('person_id')
        
    def __len__(self):
        return len(self.person_ids)
    
    def __getitem__(self, idx):
        person_id = self.person_ids[idx]
        group = self.grouped.get_group(person_id).sort_values('year')
        if len(group) < self.sequence_length:
            pad = pd.concat([group.iloc[[-1]]] * (self.sequence_length - len(group)))
            group = pd.concat([group, pad], ignore_index=True)
        
        ind_features = torch.FloatTensor(group[['age', 'ed_visits', 'medicaid_any']].values)
        
        exp_map = {'High': 2, 'Mixed': 1, 'Low': 0}
        exp_val = exp_map.get(group['expansion_group'].iloc[0], 0)
        
        sys_features = torch.zeros(len(group), 10)
        sys_features[:, 0] = exp_val
        sys_features[:, 1] = group['region'].iloc[0]
        
        targets = {
            'ed_visits': torch.FloatTensor([group['ed_visits'].iloc[0]]),
            'ed_visits_next': torch.FloatTensor([group['ed_visits'].iloc[-1]]),
            'frequent_ed': torch.FloatTensor([float(group['ed_visits'].iloc[0] >= 4)]),
            'medicaid': torch.FloatTensor([group['medicaid_any'].iloc[-1]])
        }
        
        action = torch.zeros(5)
        
        return {
            'individual_obs': ind_features,
            'system_obs': sys_features,
            'action': action,
            'targets': targets
        }

def generate_insights():
    print("Generating Novel Insights Figures...")
    
    # Load Data (Medicaid Data)
    data_path = Path("healthcare_world_model/rssm_medicaid_prepared.csv")
    if not data_path.exists():
        data_path = Path("rssm_medicaid_prepared.csv")
        
    data = pd.read_csv(data_path)
    dataset = MedicaidDataset(data.head(1000)) # Use a subset
    
    # Load Model (Medicaid Dimensions)
    device = 'cpu'
    model = HealthcareRSSM(
        individual_input_dim=3,
        system_input_dim=10,
        individual_latent_dim=16,
        system_latent_dim=8,
        action_dim=5
    ).to(device)
    
    model_path = Path("rssm_best_model.pt")
    if not model_path.exists():
        # Try parent dir
        model_path = Path("healthcare_world_model/rssm_best_model.pt")
        
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded trained model from {model_path}")
    else:
        print("Warning: Model checkpoint not found. Using initialized model.")

    model.eval()
    
    # 1. Capacity-Demand Coupling
    # Show how latent system state (capacity) correlates with individual outcomes
    print("1. Generating Capacity-Demand Coupling Figure...")
    
    # Encode all samples
    z_inds = []
    z_syss = []
    ed_probs = []
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=64)
    with torch.no_grad():
        for batch in loader:
            ind = batch['individual_obs']
            sys = batch['system_obs']
            
            enc = model.encode(ind, sys)
            z_inds.append(enc['z_individual'])
            z_syss.append(enc['z_system'])
            
            # Decode
            pred = model.decode(enc['z_individual'], enc['z_system'])
            ed_probs.append(pred['frequent_ed_prob'])
            
    z_sys_all = torch.cat(z_syss)
    ed_probs_all = torch.cat(ed_probs)
    
    # PCA on System Latent Space
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    z_sys_pca = pca.fit_transform(z_sys_all.numpy())
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=z_sys_pca.flatten(), y=ed_probs_all.flatten().numpy(), alpha=0.5) # Detach implicit in no_grad? No, flatten returns tensor
    # Actually z_sys_all is tensor. .numpy() works inside no_grad?
    # But ed_probs_all is tensor.
    # Let's be safe
    
    sns.scatterplot(x=z_sys_pca.flatten(), y=ed_probs_all.flatten().numpy(), alpha=0.5)
    plt.xlabel("System Latent State (PC1 - Capacity/Stress)")
    plt.ylabel("Predicted Frequent ED Probability")
    plt.title("Coupling: System State vs Individual Risk")
    plt.savefig("figure_coupling.png")
    plt.close()
    
    # 2. Shock Propagation (Counterfactual)
    print("2. Generating Shock Propagation Figure...")
    
    # Take one individual
    sample = dataset[0]
    ind = sample['individual_obs'].unsqueeze(0)
    sys = sample['system_obs'].unsqueeze(0)
    
    enc = model.encode(ind, sys)
    z_ind = enc['z_individual']
    z_sys = enc['z_system']
    
    # Simulate Shock (Action)
    horizon = 12
    actions_normal = torch.zeros(1, horizon, 5)
    actions_shock = torch.zeros(1, horizon, 5)
    actions_shock[:, 2, 1] = 5.0 # Huge shock at t=2 (index 1 is shock dim?)
    # Assuming dim 1 is shock.
    
    traj_normal = model.imagine_trajectory(z_ind, z_sys, actions_normal, horizon)
    traj_shock = model.imagine_trajectory(z_ind, z_sys, actions_shock, horizon)
    
    plt.figure(figsize=(10, 6))
    plt.plot(traj_normal['ed_visits'][0, :, 0].detach().numpy(), label='Baseline', marker='o')
    plt.plot(traj_shock['ed_visits'][0, :, 0].detach().numpy(), label='Shock (t=2)', marker='x', linestyle='--')
    plt.xlabel("Months")
    plt.ylabel("Predicted ED Visits")
    plt.title("Shock Propagation: Impact of System Shock on Individual Demand")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("figure_shock.png")
    plt.close()
    
    # 3. Intervention Cascades
    print("3. Generating Intervention Cascade Figure...")
    
    # Simulate Intervention (Mobile ED Unit)
    actions_intervention = torch.zeros(1, horizon, 5)
    actions_intervention[:, :, 0] = 1.0 # Continuous intervention
    
    traj_int = model.imagine_trajectory(z_ind, z_sys, actions_intervention, horizon)
    
    # Plot multiple outcomes
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    ax[0].plot(traj_normal['ed_visits'][0, :, 0].detach().numpy(), label='Baseline')
    ax[0].plot(traj_int['ed_visits'][0, :, 0].detach().numpy(), label='Intervention')
    ax[0].set_title("ED Visits")
    ax[0].legend()
    
    ax[1].plot(traj_normal['wait_time'][0, :, 0].detach().numpy(), label='Baseline')
    ax[1].plot(traj_int['wait_time'][0, :, 0].detach().numpy(), label='Intervention')
    ax[1].set_title("Wait Time (System)")
    
    ax[2].plot(traj_normal['capacity_breach_prob'][0, :, 0].detach().numpy(), label='Baseline')
    ax[2].plot(traj_int['capacity_breach_prob'][0, :, 0].detach().numpy(), label='Intervention')
    ax[2].set_title("Capacity Breach Prob")
    
    plt.suptitle("Intervention Cascades: Downstream Effects")
    plt.savefig("figure_cascade.png")
    plt.close()
    
    print("Figures generated successfully.")

if __name__ == "__main__":
    generate_insights()
