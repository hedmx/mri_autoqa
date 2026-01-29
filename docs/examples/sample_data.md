### docs/examples/sample_data.md
`markdown
# Sample Data Documentation / 示例数据文档

## 中文 (Chinese):

### 示例数据结构

examples/
├── sample_patient_001/
 │   ├── T1_scan_001/
 │    │          ├── scan.nii.gz
 │    │          └── metadata.json
 │   └── T2_scan_002/
 │              ├── scan.nii.gz
 │              └── metadata.json
└── sample_patient_002/
    └── PD_scan_001/
          ├── scan.nii.gz
          └── metadata.json
└── autoqa_results/
└── batch_report_imaging center/

### 元数据示例
``json
{

"format_version": "2.0",

"generated_date": "2026-01-01T14:42:14.388388",

"conversion_tool": "MRI_AutoQA_Converter_v2.2",

"patient_info": {

"patient_id": "P1959155",

"patient_name": "ZHANG SAN",

"patient_sex": "F",

"patient_age": "035Y"

},

"study_info": {

"study_date": "20251106",

"study_time": "134741.114000",

"study_description": "HZSDEYY^Joint",

"accession_number": "1016207719"

},

"series_info": {

"series_number": "5",

"series_description": "t1_tse_sag_p2_320",

"modality": "MR",

"protocol_name": "t1_tse_sag_p2_320"

},

"acquisition_params": {

"field_strength_t": "1.5t",

"tr_ms": 650.0,

"te_ms": 12.0,

"flip_angle_deg": 167.0,

"slice_thickness_mm": 3.0,

"pixel_spacing_mm": [

0.265625,

0.265625

],

"echo_train_length": 3,

"number_of_averages": 1.0

},

"sequence_info": {

"sequence_type": "t1",

"sequence_subtype": "fse_tse",

"manufacturer_sequence_name": "*tse2d1_3"

},

"parallel_imaging": {

"used": true,

"acceleration_factor": 2.0,

"method": "GRAPPA"

},

"anatomical_region": "spine",

"image_characteristics": {

"matrix_size": [

640,

640

],

"num_slices": 216,

"bits_allocated": 16,

"bits_stored": 12

},

"quality_flags": {

"is_valid": true,

"hard_constraint_violations": [],

"warnings": [],

"confidence_level": "PENDING",

"algorithm_route": "standard"

},

"conversion_info": {

"source_dicom_count": 216,

"conversion_date": "2026-01-01T14:42:14.389060"

}

}


## English:

### Sample Data Structure
examples/
├── sample_patient_001/
 │   ├── T1_scan_001/
 │    │          ├── scan.nii.gz
 │    │          └── metadata.json
 │   └── T2_scan_002/
 │              ├── scan.nii.gz
 │              └── metadata.json
└── sample_patient_002/
    └── PD_scan_001/
          ├── scan.nii.gz
          └── metadata.json
 └── autoqa_results/
 └── batch_report_imaging center/
### Metadata Example
``json
{

"format_version": "2.0",

"generated_date": "2026-01-01T14:42:14.388388",

"conversion_tool": "MRI_AutoQA_Converter_v2.2",

"patient_info": {

"patient_id": "P1959155",

"patient_name": "ZHANG SAN",

"patient_sex": "F",

"patient_age": "035Y"

},

"study_info": {

"study_date": "20251106",

"study_time": "134741.114000",

"study_description": "HZSDEYY^Joint",

"accession_number": "1016207719"

},

"series_info": {

"series_number": "5",

"series_description": "t1_tse_sag_p2_320",

"modality": "MR",

"protocol_name": "t1_tse_sag_p2_320"

},

"acquisition_params": {

"field_strength_t": "1.5t",

"tr_ms": 650.0,

"te_ms": 12.0,

"flip_angle_deg": 167.0,

"slice_thickness_mm": 3.0,

"pixel_spacing_mm": [

0.265625,

0.265625

],

"echo_train_length": 3,

"number_of_averages": 1.0

},

"sequence_info": {

"sequence_type": "t1",

"sequence_subtype": "fse_tse",

"manufacturer_sequence_name": "*tse2d1_3"

},

"parallel_imaging": {

"used": true,

"acceleration_factor": 2.0,

"method": "GRAPPA"

},

"anatomical_region": "spine",

"image_characteristics": {

"matrix_size": [

640,

640

],

"num_slices": 216,

"bits_allocated": 16,

"bits_stored": 12

},

"quality_flags": {

"is_valid": true,

"hard_constraint_violations": [],

"warnings": [],

"confidence_level": "PENDING",

"algorithm_route": "standard"

},

"conversion_info": {

"source_dicom_count": 216,

"conversion_date": "2026-01-01T14:42:14.389060"

}

}