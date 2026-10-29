"""Read-only C4 provenance for descriptive geometry, not policy success."""
from pathlib import Path
import hashlib,json
from target_valid_data_contract_v1 import verify_target_valid_cohort
DATASET='complete-optical-target-valid-v4'
SCOPE='Target-valid optically screened selected training cohort; descriptive geometry only, not a physical collision-free certificate or held-out policy navigation.'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_final_cohort(root,dataset):
 R=Path(root).resolve();assert dataset==DATASET
 value=verify_target_valid_cohort(R);proof=R/'report/target_valid_dataset_verification_001.json'
 assert value==json.loads(proof.read_text())['verification']
 V=json.loads((R/'dataset'/dataset/'validation_report.json').read_text())
 return {**value,'scope':SCOPE,'nonstop_actions':V['nonstop_actions'],
  'screen_exit_sha256':sha(R/'eval/optical_centerline_screen_all_selected_005/exit.json'),
  'post_verified_sha256':sha(R/'pipeline/optical_screened_dataset_v3_005/verified_dataset.json'),
  'parent_analysis_verified_sha256':sha(R/'pipeline/verified_data_analysis_002/verified_analysis.json'),
  'target_valid_verification_sha256':sha(proof),'contract_source_sha256':sha(Path(__file__)),
  'physical_legality_certified':False}
