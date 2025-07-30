import json, pandas as pd, torch, os, pickle
from torch_geometric.data import Data, Batch
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

_gene_to_protein = None
_df_interact = None
_interacting_proteins = None
_allowed_genes = None

def init_worker(gene_to_protein, df_interact, interacting_proteins, allowed_genes):
    global _gene_to_protein, _df_interact, _interacting_proteins, _allowed_genes
    _gene_to_protein = gene_to_protein
    _df_interact = df_interact
    _interacting_proteins = interacting_proteins
    _allowed_genes = allowed_genes

def build_single_patient_graph(folder):
    try:
        patient_id = os.path.basename(folder.strip('/'))
        json_expr_path = os.path.join(folder, f"{patient_id}_RNA.json")
        json_cd_path = os.path.join(folder, f"{patient_id}_CD.json")

        if not os.path.exists(json_expr_path) or not os.path.exists(json_cd_path):
            return None

        with open(json_expr_path, 'r') as f:
            gene_expr = json.load(f)
        with open(json_cd_path, 'r') as f:
            clinical_data = json.load(f)

        if "time_to_HG_recur_or_FUend" not in clinical_data:
            return None

        y_val = float(clinical_data["time_to_HG_recur_or_FUend"])

        # Map allowed gene → protein → expression
        protein_expr = {}
        gene_to_protein_used = {}
        for gene in gene_expr:
            if gene not in _allowed_genes:
                continue
            protein = _gene_to_protein.get(gene)
            if protein and protein in _interacting_proteins:
                protein_expr[protein] = gene_expr[gene]
                gene_to_protein_used[gene] = protein

        if not protein_expr:
            return None

        protein_list = list(protein_expr.keys())
        protein_to_idx = {p: i for i, p in enumerate(protein_list)}

        df_filtered = _df_interact[
            _df_interact['protein1'].isin(protein_to_idx) &
            _df_interact['protein2'].isin(protein_to_idx)
        ].copy()

        if df_filtered.empty:
            return None

        df_filtered['source'] = df_filtered['protein1'].map(protein_to_idx)
        df_filtered['target'] = df_filtered['protein2'].map(protein_to_idx)

        edge_index = torch.tensor(df_filtered[['source', 'target']].values.T, dtype=torch.long)
        edge_attr = torch.tensor(df_filtered['combined_score'].values, dtype=torch.float).unsqueeze(1) / 1000.0

        expr = torch.tensor([protein_expr[p] for p in protein_list], dtype=torch.float)
        degree = torch.bincount(edge_index.flatten(), minlength=len(protein_list)).float()

        x = torch.stack([expr, degree], dim=1)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.y = torch.tensor([y_val], dtype=torch.float)
        data.protein_to_idx = protein_to_idx
        data.idx_to_protein = {v: k for k, v in protein_to_idx.items()}
        data.gene_to_protein = gene_to_protein_used
        data.included_genes = list(gene_to_protein_used.keys())
        data.patient_id = patient_id

        return data
    except Exception as e:
        print(f"[Error] {folder}: {e}")
        return None

def build_graphs_from_json_batch_parallel(json_dirs, gene_protein_tsv, interaction_file, allowed_gene_file, max_workers=8):
    df_map = pd.read_csv(gene_protein_tsv, sep='\t')
    gene_to_protein = dict(zip(df_map['gene'], df_map['protein_id']))

    with open(allowed_gene_file, 'r') as f:
        allowed_genes = set(line.strip() for line in f if line.strip())

    all_proteins = set()
    for folder in json_dirs:
        pid = os.path.basename(folder.strip('/'))
        expr_path = os.path.join(folder, f"{pid}_RNA.json")
        if not os.path.exists(expr_path): continue
        with open(expr_path, 'r') as f:
            gene_expr = json.load(f)
        for gene in gene_expr:
            if gene in allowed_genes:
                protein = gene_to_protein.get(gene)
                if protein: all_proteins.add(protein)

    df_interact = pd.read_csv(interaction_file, delim_whitespace=True)
    df_interact = df_interact[
        df_interact['protein1'].isin(all_proteins) &
        df_interact['protein2'].isin(all_proteins)
    ].copy()
    interacting_proteins = set(df_interact['protein1']) | set(df_interact['protein2'])

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker,
                             initargs=(gene_to_protein, df_interact, interacting_proteins, allowed_genes)) as executor:
        graphs = list(tqdm(executor.map(build_single_patient_graph, json_dirs),
                           total=len(json_dirs), desc="Building graphs"))

    graphs = [g for g in graphs if g is not None]
    if not graphs:
        raise ValueError("No valid graphs were created.")

    return Batch.from_data_list(graphs)

def main():
    base_dir = "data/"
    gene_protein_tsv = "data/gene_protein_ids.tsv"
    interaction_file = "data/9606.protein.links.detailed.v12.0.txt"
    allowed_gene_file = "test.txt"
    output_path = "data/patient_graphs.pkl"

    all_patient_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and (d.startswith("3A") or d.startswith("3B"))
    ]

    print(f"[Info] Found {len(all_patient_dirs)} patient folders.")

    batch = build_graphs_from_json_batch_parallel(
        json_dirs=all_patient_dirs,
        gene_protein_tsv=gene_protein_tsv,
        interaction_file=interaction_file,
        allowed_gene_file=allowed_gene_file,
        max_workers=os.cpu_count()
    )

    with open(output_path, "wb") as f:
        pickle.dump(batch, f)

    print(f"[Done] Saved batch of {batch.num_graphs} patient graphs to {output_path}")

if __name__ == "__main__":
    main()
