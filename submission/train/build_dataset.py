"""Aim2 challenge builder: directed ligand-receptor coupling injected into real Xenium geometry.

Source: Janesick et al. 2023, Xenium FFPE Human Breast Cancer Rep1 (GEO GSM7780153): cell_feature_matrix.h5, cells.csv.gz and
the authors' supervised labels (GSM7780153_Xenium_R1_Fig1-5_supervised.csv.gz), collapsed to 8 coarse types.

Steps
  tiles  : cut the section into square tiles (side TILE_UM); keep tiles with >= MIN_CELLS labelled cells; assign tiles to
           train / validation / test by x-coordinate block (left -> right). Writes <out>/tiles.npz + tiles.csv.
  sim    : one dataset = one tile x one realisation seed. Every gene of the LR universe (LIANA consensus single-subunit pairs
           with both genes on the panel) is re-simulated under the G1-ct null (Poisson GLM on [log library, cell type] +
           smoothed Gaussian residual field matched to the gene's observed Moran's I); all other genes keep real counts.
           Injected: each directed triple (L, R, A->B) of the symmetric-eligible universe (both A->B and B->A pass the
           expression/contact filters, computed on the real counts before any simulation) independently with prob. Q_INJ. log mu(R in B) += beta * clip(z(mean L null field over A-type
           6-NN), +-4); B->A is not injected. v2: every (R, A->B) combination occurring in an eligible triple (reverse
           orientations included) gets the same form of perturbation; non-injected combinations are driven by an
           independent latent field over the type-A contacts (a receptor driven by an unobserved ligand), so receptor marginals, receptor
           spatial texture and the receiving orientation do not identify injected triples. The LR universe is
           role-separated (no gene is both ligand and receptor).
Seeds: every random draw is a function of (seed, tile, stage, gene index) via numpy SeedSequence."""
import json, os, sys, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import PoissonRegressor

SRC = '/fs/scratch/PCON0080/yimin/candidate_ccc_20261104'
XEN = f'{SRC}/results/aim2_v3/xenium'
LAB = f'{SRC}/data/janesick_xenium_rep1_GSM7780153/GSM7780153_Xenium_R1_Fig1-5_supervised.csv.gz'
LRF = f'{SRC}/results/aim2_method/inputs/liana_consensus_single.csv'
MAP = {'DCIS_1': 'Epithelial', 'DCIS_2': 'Epithelial', 'Invasive_Tumor': 'Epithelial', 'Prolif_Invasive_Tumor': 'Epithelial',
       'Myoepi_ACTA2+': 'Myoepithelial', 'Myoepi_KRT15+': 'Myoepithelial', 'CD4+_T_Cells': 'T_NK', 'CD8+_T_Cells': 'T_NK',
       'B_Cells': 'B', 'Macrophages_1': 'Myeloid', 'Macrophages_2': 'Myeloid', 'IRF7+_DCs': 'Myeloid', 'LAMP3+_DCs': 'Myeloid',
       'Stromal': 'Fibroblast', 'Endothelial': 'Endothelial', 'Perivascular-Like': 'Perivascular'}  # Mast (rare) dropped
TYPES = sorted(set(MAP.values()))
TILE_UM, MIN_CELLS, MIN_TYPE = 1000.0, 3000, 30
KNN = 6
BETAS = (0.5, 1.0, 2.0)
Q_INJ = 0.065


def tiles(out):
    import scanpy as sc
    x = sc.read_10x_h5(f'{XEN}/cell_feature_matrix.h5'); x.var_names_make_unique(); x.obs_names = x.obs_names.astype(str)
    cells = pd.read_csv(f'{XEN}/cells.csv.gz', index_col=0); cells.index = cells.index.astype(str)
    lab = pd.read_csv(LAB, index_col=0).Cluster; lab = lab[lab.index.notna()]; lab.index = lab.index.astype(int).astype(str)
    ct = lab.reindex(x.obs_names).map(MAP); keep = ct.notna().values
    genes = [g for g in x.var_names if not g.startswith(('NegControl', 'BLANK', 'antisense', 'Unassigned', 'Deprecated'))]
    x = x[keep][:, genes].copy(); ct = ct[keep].values; xy = cells.loc[x.obs_names, ['x_centroid', 'y_centroid']].values
    C = sp.csr_matrix(x.X).astype(np.int32)
    ti = np.floor((xy - xy.min(0)) / TILE_UM).astype(int); key = ti[:, 0] * 1000 + ti[:, 1]
    rows = []
    for k in np.unique(key):
        m = key == k; vc = pd.Series(ct[m]).value_counts()
        if m.sum() >= MIN_CELLS and (vc >= MIN_TYPE).sum() >= 5:
            rows.append(dict(tile=f't{k // 1000:02d}_{k % 1000:02d}', ix=k // 1000, iy=k % 1000, n=int(m.sum()),
                             n_types=int((vc >= MIN_TYPE).sum())))
    T = pd.DataFrame(rows).sort_values(['ix', 'iy']).reset_index(drop=True)
    # spatial blocks by tile column: ~60% of tiles train (left), then ~20% validation, ~20% test (right); whole columns only
    cols = sorted(T.ix.unique()); cum = T.groupby('ix').size().reindex(cols).cumsum() / len(T)
    T['split'] = T.ix.map(lambda c: 'train' if cum[c] <= 0.6 else ('validation' if cum[c] <= 0.8 else 'test'))
    arrs = dict(genes=np.array(genes, dtype='U32'), tiles=T.tile.values.astype('U16'))
    for t in T.itertuples():
        m = key == t.ix * 1000 + t.iy
        c = C[m]; arrs[f'{t.tile}/data'], arrs[f'{t.tile}/indices'], arrs[f'{t.tile}/indptr'] = c.data, c.indices, c.indptr
        arrs[f'{t.tile}/shape'] = np.array(c.shape); arrs[f'{t.tile}/cell_type'] = ct[m].astype('U16')
        arrs[f'{t.tile}/xy'] = xy[m].astype(np.float32)
    os.makedirs(out, exist_ok=True); np.savez_compressed(f'{out}/tiles.npz', **arrs); T.to_csv(f'{out}/tiles.csv', index=False)
    print(T.to_string()); print(T.split.value_counts().to_dict(), 'cells', T.groupby('split').n.sum().to_dict())


def load_tile(z, tile):
    c = sp.csr_matrix((z[f'{tile}/data'], z[f'{tile}/indices'], z[f'{tile}/indptr']), shape=tuple(z[f'{tile}/shape']))
    return c.toarray().astype(float), z[f'{tile}/cell_type'].astype(str), z[f'{tile}/xy'].astype(float)


def lr_universe(genes):
    """LIANA consensus single-subunit pairs with both genes on the panel; v2: role-separated (a gene that is a ligand in any
    pair is not used as a receptor), so a perturbed receptor never serves as an unchanged ligand."""
    gi = set(genes); lr = pd.read_csv(LRF)
    lr = lr[lr.ligand.isin(gi) & lr.receptor.isin(gi) & (lr.ligand != lr.receptor)].drop_duplicates()
    return lr[~lr.receptor.isin(set(lr.ligand))].reset_index(drop=True)


def rng_for(seed, tile_idx, *stage):
    return np.random.default_rng(np.random.SeedSequence([seed, tile_idx, *stage]))


def simulate(cnt, ctp, xy, genes, lr, seed, tile_idx):
    """Returns (sim counts int32 [n, G], truth DataFrame of injected triples, nuisance-group DataFrame, eligible DataFrame, meta)."""
    n = len(ctp); gi = {g: i for i, g in enumerate(genes)}
    vc = pd.Series(ctp).value_counts(); types = sorted(t for t in vc.index if vc[t] >= MIN_TYPE)
    keep = np.isin(ctp, types); cnt, ctp, xy = cnt[keep], ctp[keep], xy[keep]; n = len(ctp)
    _, idx = NearestNeighbors(n_neighbors=KNN + 1).fit(xy).kneighbors(xy); nbr = idx[:, 1:]
    Wm = sp.csr_matrix((np.full(idx.size, 1 / (KNN + 1)), (np.repeat(np.arange(n), KNN + 1), idx.ravel())), shape=(n, n))
    logtot = np.log(cnt.sum(1) + 1)
    X = np.column_stack([logtot] + [(ctp == t).astype(float) for t in types[1:]]); X = (X - X.mean(0)) / X.std(0)
    sf = 1e4 / np.maximum(cnt.sum(1), 1); ln = lambda y: np.log1p(y * sf); zs = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    moran = lambda v: float(zs(v) @ (Wm @ zs(v)) / n)
    det = pd.DataFrame({t: (cnt[ctp == t] > 0).mean(0) for t in types}, index=genes)
    nb = {t: (ctp == t)[nbr].sum(1) > 0 for t in types}
    ok1 = lambda l, r, a, b: det.at[l, a] >= .1 and det.at[r, b] >= .1 and ((ctp == b) & nb[a]).sum() >= 50
    elig = [(l, r, a, b) for l, r in zip(lr.ligand, lr.receptor) for a in types for b in types
            if a != b and ok1(l, r, a, b) and ok1(l, r, b, a)]
    E = pd.DataFrame(elig, columns=['ligand', 'receptor', 'source', 'target'])
    # v2b: every eligible triple is injected independently with probability Q_INJ (identical marginal for every triple, so
    # pair/type identity carries no prior); a draw with no injection is redrawn. A receiver group may receive more than one
    # injection (effects add); ligands are never modified (role-separated universe).
    r0 = rng_for(seed, tile_idx, 1)
    while True:
        hit = r0.random(len(elig)) < Q_INJ
        if hit.any(): break
    inj = [(*elig[k], float(r0.choice(BETAS))) for k in np.flatnonzero(hit)]
    inj = pd.DataFrame(inj, columns=['ligand', 'receptor', 'source', 'target', 'beta'])
    # v2c: nuisance at the (R, A->B) level: every receptor x sender x receiver combination that occurs in an eligible
    # triple and carries no injection gets its own independent latent field over the type-A contacts of B cells, with the
    # same amplitude distribution, so receptor variance in the contact cells of A->B does not reveal the sender either.
    inj_rab = set(zip(inj.receptor, inj.source, inj.target))
    combos = sorted({(r, a, b) for _, r, a, b in elig} - inj_rab)
    rd = rng_for(seed, tile_idx, 2)
    dec = [(r, a, b, float(rd.choice(BETAS))) for r, a, b in combos]
    dec = pd.DataFrame(dec, columns=['receptor', 'source', 'target', 'beta'])
    lrg = sorted(set(lr.ligand) | set(lr.receptor)); sim = cnt.copy(); field, MU, SD, KK = {}, {}, {}, []
    for g in lrg:
        rg = rng_for(seed, tile_idx, 3, gi[g]); y = cnt[:, gi[g]]
        m = PoissonRegressor(alpha=1e-4, max_iter=300).fit(X, y); mu = np.clip(m.predict(X), 1e-6, None)
        sd = np.log((y + .5) / (mu + .5)).std(); t = moran(ln(y)); best = None
        for kk in range(16):
            e = rg.normal(size=n)
            for _ in range(kk): e = Wm @ e
            f = zs(e) * sd; ys = rg.poisson(mu * np.exp(f - sd ** 2 / 2)); dd = abs(moran(ln(ys)) - t)
            if best is None or dd < best[0]: best = (dd, f, ys, kk)
        field[g], MU[g], SD[g] = best[1], mu, sd; sim[:, gi[g]] = best[2]; KK.append(best[3])

    def contact(src, tgt, fld):
        A, B = ctp == src, ctp == tgt; nbA = A[nbr]; has = B & (nbA.sum(1) > 0)
        c = np.where(has, (fld[nbr] * nbA).sum(1) / np.maximum(nbA.sum(1), 1), 0.0)
        return np.clip(c / (c[has].std() + 1e-12), -4, 4)
    EFF = {}
    for p in inj.itertuples():
        EFF[p.receptor] = EFF.get(p.receptor, 0.0) + p.beta * contact(p.source, p.target, field[p.ligand])  # adds within a group
    for i, p in enumerate(dec.itertuples()):
        rg = rng_for(seed, tile_idx, 5, gi[p.receptor], TYPES.index(p.source), TYPES.index(p.target)); e = rg.normal(size=n)
        for _ in range(int(rg.choice(KK))): e = Wm @ e
        EFF[p.receptor] = EFF.get(p.receptor, 0.0) + p.beta * contact(p.source, p.target, zs(e))
    for g, ef in EFF.items():  # all effects of one gene are added on the log scale (several senders can touch one B cell)
        rg = rng_for(seed, tile_idx, 4, gi[g])
        sim[:, gi[g]] = rg.poisson(MU[g] * np.exp(field[g] - SD[g] ** 2 / 2 + ef))
    inj['n_contact'] = [int(((ctp == b) & nb[a]).sum()) for a, b in zip(inj.source, inj.target)]
    meta = dict(n_cells=int(n), types=types, n_eligible=len(E), k_inj=len(inj), n_nuisance=len(dec))
    return sim.astype(np.int32), ctp, xy, inj, dec, E, meta


PUBLIC_SEED = 20261109
N_REAL = {'train': 6, 'validation': 3, 'test': 3}


def build(tiles_dir, out, sealed):
    """train.npz / validation.npz (with truth) and test_inputs.npz (no truth); the test truth and its secret seed go to
    <sealed>/. Per tile the real counts of all genes are stored once (cells of types with >= MIN_TYPE cells only); per dataset
    only the re-simulated LR-gene columns, the eligible triples and (train/validation) the truth."""
    import hashlib, secrets
    T = pd.read_csv(f'{tiles_dir}/tiles.csv'); z = np.load(f'{tiles_dir}/tiles.npz'); genes = list(z['genes'])
    lr = lr_universe(genes); lrg = sorted(set(lr.ligand) | set(lr.receptor)); gi = {g: i for i, g in enumerate(genes)}
    os.makedirs(out, exist_ok=True); os.makedirs(sealed, exist_ok=True)
    sf = f'{sealed}/test_seed_v2.txt'; saltf = f'{sealed}/commitment_salt_v2.txt'
    if not os.path.exists(saltf): open(saltf, 'w').write(secrets.token_hex(32))
    salt = open(saltf).read().strip().encode()
    if not os.path.exists(sf): open(sf, 'w').write(str(secrets.randbits(128)))
    test_seed = int(open(sf).read().strip()); manifest = dict(files={}, tiles=T.to_dict('records'), lr_universe=lr.to_dict('records'),
                                                               public_seed=PUBLIC_SEED, n_realisations=N_REAL)
    for split in ('train', 'validation', 'test'):
        A = dict(genes=np.array(genes, dtype='U32'), lr_genes=np.array(lrg, dtype='U32'), lr_ligand=lr.ligand.values.astype('U32'),
                 lr_receptor=lr.receptor.values.astype('U32'))
        sealed_truth, dsets = {}, []
        for t in T[T.split == split].itertuples():
            ti = int(t.Index); cnt, ctp, xy = load_tile(z, t.tile)
            vc = pd.Series(ctp).value_counts(); keep = np.isin(ctp, [k for k in vc.index if vc[k] >= MIN_TYPE])
            c = sp.csr_matrix(cnt[keep].astype(np.int32))
            for k, v in dict(data=c.data, indices=c.indices, indptr=c.indptr, shape=np.array(c.shape)).items(): A[f'{t.tile}/base_{k}'] = v
            A[f'{t.tile}/cell_type'] = ctp[keep].astype('U16'); A[f'{t.tile}/xy'] = xy[keep].astype(np.float32)
            for r in range(N_REAL[split]):
                seed = test_seed if split == 'test' else PUBLIC_SEED
                sim, ctp2, xy2, inj, dec, E, meta = simulate(cnt, ctp, xy, genes, lr, seed, ti * 100 + r)
                assert len(ctp2) == keep.sum()
                d = f'{t.tile}_r{r}'; dsets.append(d); A[f'{d}/tile'] = np.array(t.tile)
                A[f'{d}/lr_counts'] = sim[:, [gi[g] for g in lrg]].astype(np.int32)
                for col in ('ligand', 'receptor', 'source', 'target'): A[f'{d}/elig_{col}'] = E[col].values.astype('U32')
                key = {(a, b, c_, e): bb for a, b, c_, e, bb in inj[['ligand', 'receptor', 'source', 'target', 'beta']].itertuples(index=False)}
                lab = np.array([key.get(tuple(x), 0.0) for x in E.itertuples(index=False)], dtype=np.float32)  # beta of injected, 0 else
                if split == 'test': sealed_truth[d] = lab; sealed_truth[f'{d}/nuisance'] = dec.to_numpy().astype('U32')
                else:
                    A[f'{d}/label_beta'] = lab
                    for col in dec.columns: A[f'{d}/nuisance_{col}'] = dec[col].values.astype('U32')
                print(split, d, meta, 'pos', int((lab > 0).sum()), flush=True)
        A['datasets'] = np.array(dsets, dtype='U32')
        fn = {'train': 'train.npz', 'validation': 'validation.npz', 'test': 'test_inputs.npz'}[split]
        np.savez_compressed(f'{out}/{fn}', **A)
        manifest['files'][fn] = dict(datasets=dsets, sha256=hashlib.sha256(open(f'{out}/{fn}', 'rb').read()).hexdigest())
        if split == 'test':
            np.savez_compressed(f'{sealed}/test_truth.npz', **sealed_truth)
            manifest['test_truth_salted_sha256'] = hashlib.sha256(salt + open(f'{sealed}/test_truth.npz', 'rb').read()).hexdigest()
            manifest['test_seed_salted_sha256'] = hashlib.sha256(salt + open(sf, 'rb').read()).hexdigest()
            manifest['commitment_rule'] = 'sha256(salt || file bytes); the 32-byte hex salt is revealed with the truth'
    json.dump(manifest, open(f'{out}/manifest.json', 'w'), indent=1, sort_keys=True)


if __name__ == '__main__':
    if sys.argv[1] == 'tiles': tiles(sys.argv[2])
    if sys.argv[1] == 'build': build(sys.argv[2], sys.argv[3], sys.argv[4])
