from src.utils import generate_hkl_list

data_folder_name = "data"
structure_type_name = "mp20"
structure_type_path = f"{data_folder_name}/{structure_type_name}"

hkl_max_index = 10


hkl_list = generate_hkl_list(hkl_max_index)
null_hkl_id = (hkl_max_index * (hkl_max_index + 1)) * (1 + (2 * hkl_max_index + 1))