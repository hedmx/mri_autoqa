
### docs/api_documentation/core_modules.md

# Core Modules API / 核心模块API

## 中文 (Chinese):

### SingleImageSNREngine 类

class SingleImageSNREngine:
    def __init__(self, verbose: bool = True)
    def analyze_scan(self, nifti_path: Path, metadata_path: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]
    def _load_scan_data(self, nifti_path: Path, metadata_path: Path) -> Tuple[np.ndarray, Dict]
    def _build_final_result(self, **kwargs) -> Dict[str, Any]

### AlgorithmRouter 类
class AlgorithmRouter:

def route(self, metadata: Dict[str, Any]) -> Dict[str, Any]

def _extract_anatomy(self, constraint: Dict, metadata: Dict) -> str

def _extract_sequence(self, constraint: Dict, metadata: Dict) -> str

def _extract_field_strength(self, constraint: Dict, metadata: Dict) -> str

def _extract_acquisition_mode(self, constraint: Dict, metadata: Dict) -> str

### QualityMetricsCalculator 类
class QualityMetricsCalculator:

def calculate_all_metrics(self, image: np.ndarray, metadata: Dict[str, Any], signal_roi: Dict[str, Any], background_roi: Dict[str, Any]) -> Dict[str, Any]

def calculate_confidence_scores(self, image: np.ndarray, metadata: Dict[str, Any], signal_roi: Dict[str, Any], background_roi: Dict[str, Any]) -> Dict[str, Any]

def calculate_quality_scores(self, image: np.ndarray, metadata: Dict[str, Any], signal_roi: Dict[str, Any], background_roi: Dict[str, Any]) -> Dict[str, Any]

def calculate_cnr_analysis(self, image: np.ndarray, metadata: Dict[str, Any], signal_roi: Dict[str, Any], background_roi: Dict[str, Any]) -> Dict[str, Any]

## English:

### SingleImageSNREngine Class
class SingleImageSNREngine:

def __init__(self, verbose: bool = True)

def analyze_scan(self, nifti_path: Path, metadata_path: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]

def _load_scan_data(self, nifti_path: Path, metadata_path: Path) -> Tuple[np.ndarray, Dict]

def _build_final_result(self, **kwargs) -> Dict[str, Any]

### AlgorithmRouter Class
class AlgorithmRouter:

def route(self, metadata: Dict[str, Any]) -> Dict[str, Any]

def _extract_anatomy(self, constraint: Dict, metadata: Dict) -> str

def _extract_sequence(self, constraint: Dict, metadata: Dict) -> str

def _extract_field_strength(self, constraint: Dict, metadata: Dict) -> str

def _extract_acquisition_mode(self, constraint: Dict, metadata: Dict) -> str

### QualityMetricsCalculator Class
class QualityMetricsCalculator:

def calculate_all_metrics(self, image: np.ndarray, metadata: Dict[str, Any], signal_roi: Dict[str, Any], background_roi: Dict[str, Any]) -> Dict[str, Any]

def calculate_confidence_scores(self, image: np.ndarray, metadata: Dict[str, Any], signal_roi: Dict[str, Any], background_roi: Dict[str, Any]) -> Dict[str, Any]

def calculate_quality_scores(self, image: np.ndarray, metadata: Dict[str, Any], signal_roi: Dict[str, Any], background_roi: Dict[str, Any]) -> Dict[str, Any]

def calculate_cnr_analysis(self, image: np.ndarray, metadata: Dict[str, Any], signal_roi: Dict[str, Any], background_roi: Dict[str, Any]) -> Dict[str, Any]