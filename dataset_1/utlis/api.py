import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, '..', 'database')

def get_drugs(filter = None, typ = 'name'):
    """Returns the ex_drug_ids of drugs in the database,
    input drug_names or cids
    """

    if type(filter) not in [type([]), type(set([]))]:
        raise Exception("drug need to be provided in a list or set")

    drugs = pd.read_csv(os.path.join(data_file, 'drug', 'drug_id.csv'))
    drugs['CID'] = drugs['CID'].apply(lambda x: str(int(x)) if pd.notna(x) else '')

    if len(filter) > 0:
        if typ == 'cid':
            filter = list(map(str, filter))
            filtered_drugs = drugs.loc[drugs.CID.isin(filter)]
            result = pd.DataFrame({'CID': filter})
            result = result.merge(filtered_drugs, on='CID', how='left')
            missing_ids = set(filter) - set(drugs.CID)
            if missing_ids:
                print("missing CID:", missing_ids)
                with open("missing_drug_cids.txt", "w") as f:
                    for missing_id in missing_ids:
                        f.write(f"{missing_id}\n")
            return result[['DrugName', 'ex_drug_id']]

        elif typ == 'name':
            filtered_drugs = drugs.loc[drugs.DrugName.isin(filter)]
            result = pd.DataFrame({'DrugName': filter})
            result = result.merge(filtered_drugs, on='DrugName', how='left')
            missing_names = set(filter) - set(drugs.DrugName)
            if missing_names:
                print("missing names:", missing_names)
                with open("missing_drug_names.txt", "w") as f:
                    for missing_name in missing_names:
                        f.write(f"{missing_name}\n")
            return result[['DrugName', 'ex_drug_id']]

        else:
            print("Please provide drugs with tpy = name or id.")
            return None

    else:
        print('no input drugs found, return all of the available drugs.')
        return drugs[['DrugName', 'ex_drug_id']]

def get_cell_line(filter=None, typ = 'name'):
    """Returns the cell lines in the database with names, exc-ids,
        input cell line names or ach-ids
        """

    if type(filter) not in [type([]), type(set([]))]:
        raise Exception("cell line need to be provided in a list or set")

    cells = pd.read_csv(os.path.join(data_file, 'cell', 'cell_id.csv'))

    if len(filter) > 0:
        if typ == 'id':
            filtered_cells = cells.loc[cells.model_id.isin(filter)]
            result = pd.DataFrame({'model_id': filter})
            result = result.merge(filtered_cells, on='model_id', how='left')
            missing_ids = filter - set(cells.model_id)
            if missing_ids:
                print("missing ID:", missing_ids)
                with open("missing_cell_line_ids.txt", "w") as f:
                    for missing_id in missing_ids:
                        f.write(f"{missing_id}\n")
            return result[['cell_name', 'ex_cell_id']]

        elif typ == 'name':
            filtered_cells = cells.loc[cells.cell_line_name.isin(filter)]
            result = pd.DataFrame({'model_id': filter})
            result = result.merge(filtered_cells, on='model_id', how='left')
            missing_names = set(filter) - set(cells.cell_line_name)
            if missing_names:
                print("missing names:", missing_names)
                with open("missing_cell_line_names.txt", "w") as f:
                    for missing_name in missing_names:
                        f.write(f"{missing_name}\n")
            return result[['cell_name', 'ex_cell_id']]

        else:
            print("Please provide cell lines with tpy = name or id.")
            return None

    else:
        print('no input cell lines found, return all of the available cell lines.')
        return cells

def get_drug_combs(drug_comb_file, study = None, cell_line_name = None, target_name = None, avail_info = False):
    """
    Returns drug combinations based on specified criteria.
    e.g. get_drug_combs('drugcomb_cleaned_with_mean.csv', study='ONEIL', cell_line_name='A549', target_name='synergy_bliss')

    """

    drug_combs = pd.read_csv(os.path.join(data_file, 'drug_comb', drug_comb_file), na_values=["\\N", "NA", "N/A"])

    if avail_info:
        avail_study = drug_combs['study_name'].unique().tolist()
        avail_cell_lines = drug_combs['cell_line_name'].unique().tolist()
        avail_target_types = ['synergy_bliss', 'synergy_hsa', 'synergy_loewe', 'synergy_zip']
        print('Available studies are: ', avail_study)
        print('Available cell lines are: ', avail_cell_lines)
        print('Available target types are: ', avail_target_types)
        return None

    if target_name is not None:
        mask = pd.Series([True] * len(drug_combs))
        if study is not None:
            mask &= (drug_combs['study_name'] == study)
        if cell_line_name is not None:
            mask &= (drug_combs['cell_line_name'] == cell_line_name)
        filtered_drug_combs = drug_combs[mask]
        print('drug_combs_shape:', filtered_drug_combs.shape)
        filtered_drug_combs.dropna(subset=['drug_row', 'drug_col', 'cell_line_name', 'study_name', target_name], inplace=True)
        print('return_drug_combs_dropna_shape:', filtered_drug_combs.shape)

        return filtered_drug_combs[['drug_row', 'drug_col', 'cell_line_name', 'study_name', target_name, 'block_id']]

    else:
        print("no designated target name, return all types of synergy score.")
        mask = pd.Series([True] * len(drug_combs))
        if study is not None:
            mask &= (drug_combs['study_name'] == study)
        if cell_line_name is not None:
            mask &= (drug_combs['cell_line_name'] == cell_line_name)
        filtered_drug_combs = drug_combs[mask]
        print('drug_combs_shape:', filtered_drug_combs.shape)
        filtered_drug_combs.dropna(subset=['drug_row', 'drug_col', 'cell_line_name', 'study_name'], inplace=True)
        print('return_drug_combs_dropna_shape:', filtered_drug_combs.shape)

        return filtered_drug_combs



