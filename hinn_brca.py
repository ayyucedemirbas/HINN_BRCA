import os
import sys
import random
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.utils import resample
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F

import captum
from captum.attr import DeepLift
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc

import warnings
warnings.filterwarnings('ignore')


def load_and_process_brca_data():
    try:
        df = pd.read_csv('data.csv')
        
        if df.isnull().sum().sum() > 0:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        unique_values = df['vital.status'].unique()
        
        if set(unique_values).issubset({0, 1}):
            pass
        elif set(unique_values).issubset({'Alive', 'Dead', 'alive', 'dead'}):
            df['vital.status'] = df['vital.status'].str.lower().map({'alive': 0, 'dead': 1})
        elif set(unique_values).issubset({'Living', 'Deceased', 'living', 'deceased'}):
            df['vital.status'] = df['vital.status'].str.lower().map({'living': 0, 'deceased': 1})
        else:
            le = LabelEncoder()
            df['vital.status'] = le.fit_transform(df['vital.status'])
        
        
        df_majority = df[df['vital.status'] == 0]
        df_minority = df[df['vital.status'] == 1]
        
        df_minority_upsampled = resample(df_minority,
                                         replace=True,
                                         n_samples=len(df_majority),
                                         random_state=42)
        
        df_balanced = pd.concat([df_majority, df_minority_upsampled])


        
        mu_cols = [col for col in df_balanced.columns if col.startswith('mu_')]
        cn_cols = [col for col in df_balanced.columns if col.startswith('cn_')]
        rs_cols = [col for col in df_balanced.columns if col.startswith('rs_')]
        pp_cols = [col for col in df_balanced.columns if col.startswith('pp_')]
        
        data = df_balanced.copy()
        
        data = data.rename(columns={'vital.status': 'vital_status_label'})
        
        return data, mu_cols, cn_cols, rs_cols, pp_cols
        
    except FileNotFoundError:
        print("Error: data.csv file not found.")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None, None, None


def create_sparse_connectivity_matrices(mu_cols, cn_cols, rs_cols, pp_cols, 
                                        connectivity_ratio=0.3, seed=42):

    #Mutation (mu_) -> Copy Number (cn_) -> RNA-seq (rs_) -> Protein (pp_)

    np.random.seed(seed)
    
    # Layer 1: Mutation (mu_) -> Copy Number (cn_)
    n_mu = len(mu_cols)
    n_cn = len(cn_cols)
    
    sparse_mu_cn = np.random.rand(n_mu, n_cn) < connectivity_ratio
    sparse_mu_cn_df = pd.DataFrame(
        sparse_mu_cn.astype(int),
        index=mu_cols,
        columns=cn_cols
    )
    
    # Layer 2: Copy Number (cn_) -> RNA-seq (rs_)
    n_rs = len(rs_cols)
    
    sparse_cn_rs = np.random.rand(n_cn, n_rs) < connectivity_ratio
    sparse_cn_rs_df = pd.DataFrame(
        sparse_cn_rs.astype(int),
        index=cn_cols,
        columns=rs_cols
    )
    
    # Layer 3: RNA-seq (rs_) -> Protein (pp_)
    n_pp = len(pp_cols)
    
    sparse_rs_pp = np.random.rand(n_rs, n_pp) < min(0.4, connectivity_ratio + 0.1)
    sparse_rs_pp_df = pd.DataFrame(
        sparse_rs_pp.astype(int),
        index=rs_cols,
        columns=pp_cols
    )
    
    print(f"  Layer 1 (Mutation -> Copy Number): {sparse_mu_cn_df.shape}, connections: {sparse_mu_cn_df.sum().sum()}")
    print(f"  Layer 2 (Copy Number -> RNA-seq): {sparse_cn_rs_df.shape}, connections: {sparse_cn_rs_df.sum().sum()}")
    print(f"  Layer 3 (RNA-seq -> Protein): {sparse_rs_pp_df.shape}, connections: {sparse_rs_pp_df.sum().sum()}")
    
    return sparse_mu_cn_df, sparse_cn_rs_df, sparse_rs_pp_df


class PrimaryInputLayer(nn.Module):
    def __init__(self, units, output_dim, activation="sigmoid", mask=None):
        super().__init__()
        self.units = units
        self.output_dim = output_dim

        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.w = nn.Parameter(torch.empty(units, output_dim))
        self.b = nn.Parameter(torch.zeros(output_dim))
        nn.init.xavier_normal_(self.w)

        if mask is None:
            raise ValueError("mask tensor is required")
        self.register_buffer("mask", mask.float())

    def forward(self, x):
        masked_w = self.w * self.mask
        out = x @ masked_w + self.b
        return self.activation(out)


class SecondaryInputLayer(nn.Module):
    def __init__(self, units):
        super().__init__()
        self.units = units
        self.register_buffer("mask", torch.eye(units))
        self.w = nn.Parameter(torch.empty(units, units))
        nn.init.xavier_normal_(self.w)

    def forward(self, x):
        masked_w = self.w * self.mask
        return x @ masked_w


class MultiplicationInputLayer(nn.Module):
    def __init__(self, units, activation="sigmoid"):
        super().__init__()
        self.units = units

        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.b = nn.Parameter(torch.zeros(units))
        nn.init.xavier_normal_(self.b.unsqueeze(0))

    def forward(self, x):
        return self.activation(x + self.b)


class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return [input[idx] for input in self.inputs], self.targets[idx]



class EarlyStopping:
    def __init__(self, patience=50, delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.delta = delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()

        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"Early stopping triggered. Best val_loss = {self.best_loss:.4f}")
            if self.restore_best_weights and self.best_model_state is not None:
                model.load_state_dict(self.best_model_state)
            return True

        return False




class HINN_Classifier(nn.Module):
    #Mutation (mu_) -> Copy Number (cn_) -> RNA-seq (rs_) -> Protein (pp_) -> Prediction
    def __init__(
        self,
        mu_dim,
        cn_dim,
        rs_dim,
        pp_dim,
        sparse_mu_cn_tensor,
        sparse_cn_rs_tensor,
        sparse_rs_pp_tensor,
        dense_nodes_1=128,
        drop_rate=0.7,
        activation_function="sigmoid",
    ):
        super().__init__()

        # First block: Mutation (mu_) -> Copy Number (cn_)
        self.primary1 = PrimaryInputLayer(
            units=mu_dim,
            output_dim=cn_dim,
            activation=activation_function,
            mask=sparse_mu_cn_tensor,
        )
        self.secondary1 = SecondaryInputLayer(units=cn_dim)
        self.mult1 = MultiplicationInputLayer(
            units=cn_dim,
            activation=activation_function,
        )
        self.mu_fc = nn.Linear(mu_dim, 20)

        # Second block: Copy Number (cn_) -> RNA-seq (rs_)
        self.primary2 = PrimaryInputLayer(
            units=cn_dim,
            output_dim=rs_dim,
            activation=activation_function,
            mask=sparse_cn_rs_tensor,
        )
        self.secondary2 = SecondaryInputLayer(units=rs_dim)
        self.mult2 = MultiplicationInputLayer(
            units=rs_dim,
            activation=activation_function,
        )
        self.mid_fc = nn.Linear(cn_dim + 20, 20)

        # Third block: RNA-seq (rs_) -> Protein (pp_)
        self.primary3 = PrimaryInputLayer(
            units=rs_dim,
            output_dim=pp_dim,
            activation=activation_function,
            mask=sparse_rs_pp_tensor,
        )
        self.mid_fc2 = nn.Linear(rs_dim + 20, 20)

        # Dense layers
        custom_input_dim = pp_dim + 20

        self.bn1 = nn.BatchNorm1d(custom_input_dim)
        self.fc1 = nn.Linear(custom_input_dim, dense_nodes_1)
        self.drop1 = nn.Dropout(drop_rate)

        self.bn2 = nn.BatchNorm1d(dense_nodes_1)
        self.fc2 = nn.Linear(dense_nodes_1, dense_nodes_1)
        self.drop2 = nn.Dropout(drop_rate)

        self.bn3 = nn.BatchNorm1d(dense_nodes_1)
        self.fc3 = nn.Linear(dense_nodes_1, dense_nodes_1)
        self.drop3 = nn.Dropout(drop_rate)

        self.bn4 = nn.BatchNorm1d(dense_nodes_1)
        self.fc4 = nn.Linear(dense_nodes_1, dense_nodes_1)
        self.drop4 = nn.Dropout(drop_rate)

        self.dense_fourth = nn.Linear(dense_nodes_1, 20)
        
        self.bn_final = nn.BatchNorm1d(20 + pp_dim)
        self.fc_final = nn.Linear(20 + pp_dim, dense_nodes_1)
        self.drop_final = nn.Dropout(drop_rate)

        self.out = nn.Linear(dense_nodes_1, 1)

        self.activation_function = activation_function

    def _nonlin(self, x):
        return torch.sigmoid(x)

    def forward(self, mu, cn, rs, pp):
        # First block: Mutation -> Copy Number
        primary1 = self.primary1(mu)
        secondary1 = self.secondary1(cn)
        mult_res1 = primary1 * secondary1
        mult1 = self.mult1(mult_res1)

        mu_fc = self._nonlin(self.mu_fc(mu))
        out2 = torch.cat([mult1, mu_fc], dim=1)

        # Second block: Copy Number -> RNA-seq
        primary2 = self.primary2(mult1)
        secondary2 = self.secondary2(rs)

        eps = 1e-6
        denom = primary2.clone()
        denom = torch.where(denom.abs() < eps, eps * torch.ones_like(denom), denom)
        div_res1 = secondary2 / denom
        div_res1 = torch.clamp(div_res1, -1e6, 1e6)
        mult2 = self.mult2(div_res1)

        mid_fc = self._nonlin(self.mid_fc(out2))
        out3 = torch.cat([mult2, mid_fc], dim=1)

        # Third block: RNA-seq -> Protein
        primary3 = self.primary3(mult2)
        mid_fc2 = self._nonlin(self.mid_fc2(out3))
        out4 = torch.cat([primary3, mid_fc2], dim=1)

        # Dense stack
        x = self.bn1(out4)
        x = self._nonlin(self.fc1(x))
        x = self.drop1(x)

        x = self.bn2(x)
        x = self._nonlin(self.fc2(x))
        x = self.drop2(x)

        x = self.bn3(x)
        x = self._nonlin(self.fc3(x))
        x = self.drop3(x)

        x = self.bn4(x)
        x = self._nonlin(self.fc4(x))
        x = self.drop4(x)

        dense_fourth = self._nonlin(self.dense_fourth(x))
        
        protein_concat = torch.cat([dense_fourth, pp], dim=1)

        x = self.bn_final(protein_concat)
        x = self._nonlin(self.fc_final(x))
        x = self.drop_final(x)

        out = torch.sigmoid(self.out(x))
        return out


def train_model_torch(model, train_loader, val_loader, device="cpu",
                      lr=1e-3, epochs=1000, patience=50):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopper = EarlyStopping(patience=patience, delta=0.0, restore_best_weights=True)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs = [x.to(device).float() for x in inputs]
            targets = targets.to(device).float().unsqueeze(1)

            if inputs[0].size(0) == 1:
                model.eval()
                with torch.no_grad():
                    outputs = model(*inputs)
                model.train()
            else:
                optimizer.zero_grad()
                outputs = model(*inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * targets.size(0)
                
                predicted = (outputs > 0.5).float()
                train_correct += (predicted == targets).sum().item()
                train_total += targets.size(0)

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total if train_total > 0 else 0

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = [x.to(device).float() for x in inputs]
                targets = targets.to(device).float().unsqueeze(1)
                outputs = model(*inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0)
                
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total if val_total > 0 else 0

        print(f"Epoch {epoch+1:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if early_stopper(val_loss, model):
            print(f"Stopping at epoch {epoch+1}")
            break

    return model

def evaluate_model_torch(model, test_loader, device="cpu"):
    model.eval()
    model.to(device)

    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = [x.to(device).float() for x in inputs]
            targets = targets.to(device).float()

            probs = model(*inputs).squeeze()
            preds = (probs > 0.5).float()

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    print(classification_report(y_true, y_pred, target_names=['Alive', 'Deceased']))
    
    print(confusion_matrix(y_true, y_pred))

    return {
        "accuracy": accuracy,
        "auc": auc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob
    }


def interpret_model(model, test_inputs, baselines, device="cpu"):
    model.eval()
    model.to(device)

    test_inputs = tuple(t.to(device) for t in test_inputs)
    baselines = tuple(b.to(device) for b in baselines)

    explainer = DeepLift(model)
    attributions = explainer.attribute(
        test_inputs,
        baselines=baselines,
        return_convergence_delta=False,
    )
    return attributions


def export_attributions(attributions, feature_names, save_path_prefix):
    modality_names = ['mutation', 'copynumber', 'rnaseq', 'protein']
    for i, name in enumerate(modality_names):
        attr_cpu = attributions[i].detach().cpu().numpy()
        df = pd.DataFrame(attr_cpu, columns=feature_names[i])
        df.to_csv(f"{save_path_prefix}_{name}.csv", index=False)
        print(f"Saved attributions to {save_path_prefix}_{name}.csv")


def plot_feature_importance(attributions, feature_names, top_k=20):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    modality_names = ['Mutations', 'Copy Number', 'RNA-seq', 'Protein']
    
    for i, (attr, names, title) in enumerate(zip(attributions, feature_names, modality_names)):
        importance = attr.abs().mean(dim=0).detach().cpu().numpy()
        top_idx = np.argsort(-importance)[:top_k]
        
        top_features = [names[j] for j in top_idx]
        top_importance = importance[top_idx]
        
        axes[i].barh(range(len(top_features)), top_importance)
        axes[i].set_yticks(range(len(top_features)))
        axes[i].set_yticklabels(top_features, fontsize=8)
        axes[i].set_xlabel('Mean Absolute Attribution')
        axes[i].set_title(f'Top {top_k} Important {title} Features')
        axes[i].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()


def filter_matrices_by_top_features(mu_list, cn_list, rs_list, pp_list,
                                     sparse_mu_cn, sparse_cn_rs, sparse_rs_pp):

    # Filter Mutation -> Copy Number connections
    subset_mu_cn = sparse_mu_cn.loc[mu_list, cn_list]
    subset_mu_cn = subset_mu_cn.loc[
        subset_mu_cn.any(axis=1), 
        subset_mu_cn.any(axis=0)
    ]
    
    # Filter Copy Number -> RNA-seq connections
    subset_cn_rs = sparse_cn_rs.loc[cn_list, rs_list]
    subset_cn_rs = subset_cn_rs.loc[
        subset_cn_rs.any(axis=1), 
        subset_cn_rs.any(axis=0)
    ]
    
    # Filter RNA-seq -> Protein connections
    subset_rs_pp = sparse_rs_pp.loc[rs_list, pp_list]
    subset_rs_pp = subset_rs_pp.loc[
        subset_rs_pp.index.isin(subset_cn_rs.columns)
    ]
    subset_rs_pp = subset_rs_pp.loc[
        subset_rs_pp.any(axis=1), 
        subset_rs_pp.any(axis=0)
    ]
    
    return subset_mu_cn, subset_cn_rs, subset_rs_pp


def summarize_connections(*matrices):
    connection_counts = [int(matrix.sum().sum()) for matrix in matrices]
    labels = ["Mutation -> Copy Number", "Copy Number -> RNA-seq", "RNA-seq -> Protein"]
    for label, count in zip(labels, connection_counts):
        print(f"  {label}: {count} connections")


def build_edge_list(subset_mu_cn, subset_cn_rs, subset_rs_pp):
    # Mutation -> Copy Number edges
    edges_mu_cn = (
        subset_mu_cn[subset_mu_cn == 1]
        .stack()
        .reset_index()
    )
    edges_mu_cn.columns = ["source", "target", "value"]
    edges_mu_cn["layer"] = "mu_cn"
    
    edges_cn_rs = (
        subset_cn_rs[subset_cn_rs == 1]
        .stack()
        .reset_index()
    )
    edges_cn_rs.columns = ["source", "target", "value"]
    edges_cn_rs["layer"] = "cn_rs"
    
    edges_rs_pp = (
        subset_rs_pp[subset_rs_pp == 1]
        .stack()
        .reset_index()
    )
    edges_rs_pp.columns = ["source", "target", "value"]
    edges_rs_pp["layer"] = "rs_pp"
    
    edges_all = pd.concat(
        [edges_mu_cn, edges_cn_rs, edges_rs_pp],
        ignore_index=True,
    )
    
    edges_all["value"] = 1
    
    return edges_all


def plot_sankey_from_edges(edges_all):
    edges_all_filtered = edges_all.copy()
    
    nodes = pd.unique(edges_all_filtered[["source", "target"]].values.ravel())
    
    mutations = [node for node in nodes if node.startswith("mu_")]
    copy_numbers = [node for node in nodes if node.startswith("cn_")]
    rna_seqs = [node for node in nodes if node.startswith("rs_")]
    proteins = [node for node in nodes if node.startswith("pp_")]
    
    ordered_nodes = mutations + copy_numbers + rna_seqs + proteins
    
    node_indices = {name: i for i, name in enumerate(ordered_nodes)}
    
    edges_all_filtered = edges_all_filtered[
        edges_all_filtered["source"].isin(ordered_nodes)
        & edges_all_filtered["target"].isin(ordered_nodes)
    ].copy()
    
    edges_all_filtered["source_index"] = edges_all_filtered["source"].map(node_indices)
    edges_all_filtered["target_index"] = edges_all_filtered["target"].map(node_indices)
    
    node_positions_x = [
        0.0 if node in mutations
        else 0.33 if node in copy_numbers
        else 0.66 if node in rna_seqs
        else 0.99
        for node in ordered_nodes
    ]
    
    node_colors = []
    for node in ordered_nodes:
        if node in mutations:
            node_colors.append('rgba(255, 99, 71, 0.8)')
        elif node in copy_numbers:
            node_colors.append('rgba(70, 130, 180, 0.8)')
        elif node in rna_seqs:
            node_colors.append('rgba(60, 179, 113, 0.8)')
        else:  # proteins
            node_colors.append('rgba(218, 112, 214, 0.8)')
    
    node_labels = [node.split('_', 1)[1] if '_' in node else node for node in ordered_nodes]
    
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors,
            x=node_positions_x,
        ),
        link=dict(
            source=edges_all_filtered["source_index"],
            target=edges_all_filtered["target_index"],
            value=edges_all_filtered["value"],
            color='rgba(150, 150, 150, 0.3)',
        ),
    ))
    
    fig.update_layout(
        title_text="BRCA Multi-Omics Hierarchical Network (4 Layers): Mutation -> Copy Number -> RNA-seq -> Protein",
        font_size=12,
        height=1200,
        width=1800,
    )
    
    fig.write_html('brca_sankey_diagram.html')
    fig.show()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data, mu_cols, cn_cols, rs_cols, pp_cols = load_and_process_brca_data()
    
    if data is None:
        print("Failed to load data. Exiting.")
        return

    sparse_mu_cn, sparse_cn_rs, sparse_rs_pp = create_sparse_connectivity_matrices(
        mu_cols, cn_cols, rs_cols, pp_cols, connectivity_ratio=0.3, seed=42
    )

    y = data["vital_status_label"]
    X = data.drop(columns=["vital_status_label"])

    X_train_int, X_test_df, y_train_int, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_train_int, y_train_int, test_size=0.2, random_state=42, stratify=y_train_int
    )

    X_train_mu = X_train_df[[col for col in X_train_df.columns if col.startswith("mu_")]].values
    X_train_cn = X_train_df[[col for col in X_train_df.columns if col.startswith("cn_")]].values
    X_train_rs = X_train_df[[col for col in X_train_df.columns if col.startswith("rs_")]].values
    X_train_pp = X_train_df[[col for col in X_train_df.columns if col.startswith("pp_")]].values

    X_train_list = [
        torch.tensor(X_train_mu, dtype=torch.float32),
        torch.tensor(X_train_cn, dtype=torch.float32),
        torch.tensor(X_train_rs, dtype=torch.float32),
        torch.tensor(X_train_pp, dtype=torch.float32),
    ]
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)

    X_val_mu = X_val_df[[col for col in X_val_df.columns if col.startswith("mu_")]].values
    X_val_cn = X_val_df[[col for col in X_val_df.columns if col.startswith("cn_")]].values
    X_val_rs = X_val_df[[col for col in X_val_df.columns if col.startswith("rs_")]].values
    X_val_pp = X_val_df[[col for col in X_val_df.columns if col.startswith("pp_")]].values

    X_val_list = [
        torch.tensor(X_val_mu, dtype=torch.float32),
        torch.tensor(X_val_cn, dtype=torch.float32),
        torch.tensor(X_val_rs, dtype=torch.float32),
        torch.tensor(X_val_pp, dtype=torch.float32),
    ]
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32)

    X_test_mu = X_test_df[[col for col in X_test_df.columns if col.startswith("mu_")]].values
    X_test_cn = X_test_df[[col for col in X_test_df.columns if col.startswith("cn_")]].values
    X_test_rs = X_test_df[[col for col in X_test_df.columns if col.startswith("rs_")]].values
    X_test_pp = X_test_df[[col for col in X_test_df.columns if col.startswith("pp_")]].values

    X_test_list = [
        torch.tensor(X_test_mu, dtype=torch.float32),
        torch.tensor(X_test_cn, dtype=torch.float32),
        torch.tensor(X_test_rs, dtype=torch.float32),
        torch.tensor(X_test_pp, dtype=torch.float32),
    ]
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32)

    train_dataset = CustomDataset(X_train_list, y_train_t)
    val_dataset = CustomDataset(X_val_list, y_val_t)
    test_dataset = CustomDataset(X_test_list, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    sparse_mu_cn_tensor = torch.tensor(sparse_mu_cn.values, dtype=torch.float32)
    sparse_cn_rs_tensor = torch.tensor(sparse_cn_rs.values, dtype=torch.float32)
    sparse_rs_pp_tensor = torch.tensor(sparse_rs_pp.values, dtype=torch.float32)

    
    mu_dim = X_train_mu.shape[1]
    cn_dim = X_train_cn.shape[1]
    rs_dim = X_train_rs.shape[1]
    pp_dim = X_train_pp.shape[1]

    print(f"  Mutation (mu_): {mu_dim}")
    print(f"  Copy Number (cn_): {cn_dim}")
    print(f"  RNA-seq (rs_): {rs_dim}")
    print(f"  Protein (pp_): {pp_dim}")

    model = HINN_Classifier(
        mu_dim=mu_dim,
        cn_dim=cn_dim,
        rs_dim=rs_dim,
        pp_dim=pp_dim,
        sparse_mu_cn_tensor=sparse_mu_cn_tensor,
        sparse_cn_rs_tensor=sparse_cn_rs_tensor,
        sparse_rs_pp_tensor=sparse_rs_pp_tensor,
        dense_nodes_1=128,
        drop_rate=0.7,
        activation_function="sigmoid",
    )
    
    model = train_model_torch(
        model,
        train_loader,
        val_loader,
        device=device,
        lr=1e-3,
        epochs=1000,
        patience=50,
    )

    eval_results = evaluate_model_torch(model, test_loader, device=device)
    print(f"\nTest Accuracy: {eval_results['accuracy']:.4f}")
    print(f"Test AUC: {eval_results['auc']:.4f}")

    
    test_inputs = tuple(
        torch.tensor(arr, dtype=torch.float32, requires_grad=True).to(device)
        for arr in [X_test_mu, X_test_cn, X_test_rs, X_test_pp]
    )

    baselines = tuple(
        torch.tensor(arr.mean(axis=0), dtype=torch.float32)
        .unsqueeze(0)
        .expand_as(torch.tensor(arr, dtype=torch.float32))
        .to(device)
        for arr in [X_test_mu, X_test_cn, X_test_rs, X_test_pp]
    )

    attributions = interpret_model(model, test_inputs, baselines, device=device)

    feature_names = [
        [col for col in X_train_df.columns if col.startswith("mu_")],
        [col for col in X_train_df.columns if col.startswith("cn_")],
        [col for col in X_train_df.columns if col.startswith("rs_")],
        [col for col in X_train_df.columns if col.startswith("pp_")]
    ]
    export_attributions(attributions, feature_names, "BRCA_vital_status")

    try:
        plot_feature_importance(attributions, feature_names, top_k=20)
    except Exception as e:
        print(f"Could not generate feature importance plots: {e}")
    
    mu_importance = attributions[0].abs().mean(dim=0).detach().cpu().numpy()
    cn_importance = attributions[1].abs().mean(dim=0).detach().cpu().numpy()
    rs_importance = attributions[2].abs().mean(dim=0).detach().cpu().numpy()
    pp_importance = attributions[3].abs().mean(dim=0).detach().cpu().numpy()
    
    TOP_MU = 15
    TOP_CN = 30
    TOP_RS = 30
    TOP_PP = 20
    
    top_mu_idx = np.argsort(-mu_importance)[:TOP_MU]
    top_cn_idx = np.argsort(-cn_importance)[:TOP_CN]
    top_rs_idx = np.argsort(-rs_importance)[:TOP_RS]
    top_pp_idx = np.argsort(-pp_importance)[:TOP_PP]
    
    mu_list = [feature_names[0][i] for i in top_mu_idx]
    cn_list = [feature_names[1][i] for i in top_cn_idx]
    rs_list = [feature_names[2][i] for i in top_rs_idx]
    pp_list = [feature_names[3][i] for i in top_pp_idx]
    
    try:
        subset_mu_cn, subset_cn_rs, subset_rs_pp = filter_matrices_by_top_features(
            mu_list, cn_list, rs_list, pp_list,
            sparse_mu_cn, sparse_cn_rs, sparse_rs_pp
        )
        
        summarize_connections(subset_mu_cn, subset_cn_rs, subset_rs_pp)
        
        edges_all = build_edge_list(subset_mu_cn, subset_cn_rs, subset_rs_pp)
        
        if len(edges_all) > 0:
            plot_sankey_from_edges(edges_all)
        else:
            print("No connections found for Sankey diagram")
            
    except Exception as e:
        print(f"Could not generate Sankey diagram: {e}")
        import traceback
        traceback.print_exc()


    print(f"  Accuracy: {eval_results['accuracy']:.4f}")
    print(f"  AUC Score: {eval_results['auc']:.4f}")


if __name__ == "__main__":
    main()
