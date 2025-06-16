from ._tokenizer import CIFTokenizer

try:
    from ._metrics import (
        bond_length_reasonableness_score,
        is_space_group_consistent,
        is_atom_site_multiplicity_consistent,
        is_formula_consistent,
        is_sensible,
        is_valid,
    )
except Exception:  # optional dependency "pymatgen" may be missing
    bond_length_reasonableness_score = None
    is_space_group_consistent = None
    is_atom_site_multiplicity_consistent = None
    is_formula_consistent = None
    is_sensible = None
    is_valid = None

try:
    from ._model import (
        GPT,
        GPTConfig,
    )
except Exception:  # optional dependency "torch" may be missing
    GPT = None
    GPTConfig = None

try:
    from ._utils import (
        array_split,
        add_atomic_props_block,
        embeddings_from_csv,
        extract_data_formula,
        extract_formula_nonreduced,
        extract_formula_units,
        extract_numeric_property,
        extract_space_group_symbol,
        extract_volume,
        get_atomic_props_block,
        get_atomic_props_block_for_formula,
        get_unit_cell_volume,
        remove_atom_props_block,
        replace_data_formula_with_nonreduced_formula,
        replace_symmetry_operators,
        round_numbers,
        semisymmetrize_cif,
    )
except Exception:
    array_split = None
    add_atomic_props_block = None
    embeddings_from_csv = None
    extract_data_formula = None
    extract_formula_nonreduced = None
    extract_formula_units = None
    extract_numeric_property = None
    extract_space_group_symbol = None
    extract_volume = None
    get_atomic_props_block = None
    get_atomic_props_block_for_formula = None
    get_unit_cell_volume = None
    remove_atom_props_block = None
    replace_data_formula_with_nonreduced_formula = None
    replace_symmetry_operators = None
    round_numbers = None
    semisymmetrize_cif = None

try:
    from ._scorer import (
        CIFScorer,
        RandomScorer,
        ZMQScorer,
    )
except Exception:
    CIFScorer = None
    RandomScorer = None
    ZMQScorer = None

try:
    from ._configuration import parse_config
except Exception:
    parse_config = None

try:
    from ._mcts import (
        ContextSensitiveTreeBuilder,
        GreedySelector,
        MCTSSampler,
        MCTSEvaluator,
        PUCTSelector,
        UCTSelector,
    )
except Exception:
    ContextSensitiveTreeBuilder = None
    GreedySelector = None
    MCTSSampler = None
    MCTSEvaluator = None
    PUCTSelector = None
    UCTSelector = None
