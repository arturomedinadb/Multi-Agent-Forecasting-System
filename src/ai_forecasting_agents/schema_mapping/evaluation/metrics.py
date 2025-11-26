"""Custom evaluation metrics for schema mapping agents using DeepEval framework."""
from typing import Dict, List, Any, Optional
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class FieldCoverageMetric(BaseMetric):
    """
    Metric to evaluate how many required target schema fields were mapped.
    
    Score = (number of covered fields) / (total required fields)
    """
    
    def __init__(
        self,
        required_fields: List[str],
        threshold: float = 0.9,
        strict_mode: bool = False,
        async_mode: bool = False,
    ):
        self.required_fields = required_fields
        self.threshold = threshold
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.score: Optional[float] = None
        self.success: Optional[bool] = None
        self.score_breakdown: Dict[str, Any] = {}
        self.reason: Optional[str] = None
        self.error: Optional[str] = None
    
    def measure(self, test_case: LLMTestCase) -> float:
        """Calculate field coverage score."""
        try:
            # Extract mapped columns from test case metadata
            metadata = test_case.additional_metadata or {}
            dataset_columns = metadata.get("dataset_columns", [])
            
            # Calculate coverage
            required_set = set(self.required_fields)
            covered_set = set(dataset_columns).intersection(required_set)
            missing_fields = list(required_set - covered_set)
            
            if len(self.required_fields) == 0:
                self.score = 1.0
            else:
                self.score = len(covered_set) / len(self.required_fields)
            
            # Store detailed breakdown
            self.score_breakdown = {
                "covered_fields": sorted(list(covered_set)),
                "missing_fields": sorted(missing_fields),
                "required_total": len(self.required_fields),
                "covered_count": len(covered_set),
                "coverage_ratio": self.score,
            }
            
            # Determine success
            self.success = self.score >= self.threshold
            
            # Generate reason
            if self.success:
                self.reason = f"Field coverage of {self.score:.2%} meets threshold {self.threshold:.2%}"
            else:
                self.reason = (
                    f"Field coverage of {self.score:.2%} below threshold {self.threshold:.2%}. "
                    f"Missing {len(missing_fields)} fields: {', '.join(missing_fields[:5])}"
                    + ("..." if len(missing_fields) > 5 else "")
                )
            
            return self.score
            
        except Exception as e:
            self.error = str(e)
            self.score = 0.0
            self.success = False
            raise
    
    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async version of measure."""
        return self.measure(test_case)
    
    def is_successful(self) -> bool:
        """Check if metric evaluation was successful."""
        if self.error is not None:
            self.success = False
        return self.success if self.success is not None else False
    
    @property
    def __name__(self):
        return "Field Coverage"


class TypeCompatibilityMetric(BaseMetric):
    """
    Metric to evaluate if mapped column types are compatible with target schema types.
    
    Score = (number of compatible fields) / (total inspected fields)
    """
    
    def __init__(
        self,
        expected_types: Dict[str, str],
        threshold: float = 1.0,
        strict_mode: bool = True,
        async_mode: bool = False,
    ):
        self.expected_types = expected_types
        self.threshold = threshold
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.score: Optional[float] = None
        self.success: Optional[bool] = None
        self.score_breakdown: Dict[str, Any] = {}
        self.reason: Optional[str] = None
        self.error: Optional[str] = None
    
    def _is_type_compatible(self, observed_type: str, expected_type: str) -> bool:
        """Check if observed type is compatible with expected type."""
        # Normalize types
        observed = observed_type.lower()
        expected = expected_type.lower()
        
        # Direct match
        if observed == expected:
            return True
        
        # Compatible type mappings
        compatible_mappings = {
            "string": ["str", "object", "text", "varchar"],
            "integer": ["int", "int32", "int64", "long"],
            "number": ["float", "float32", "float64", "double", "numeric", "int", "int32", "int64"],
            "boolean": ["bool", "bit"],
            "array": ["list"],
        }
        
        for target_type, compatible_types in compatible_mappings.items():
            if expected == target_type and observed in compatible_types:
                return True
            if expected in compatible_types and observed == target_type:
                return True
        
        return False
    
    def measure(self, test_case: LLMTestCase) -> float:
        """Calculate type compatibility score."""
        try:
            # Extract metadata
            metadata = test_case.additional_metadata or {}
            mapping_plan = metadata.get("mapping_plan", [])
            dataset_dtypes = metadata.get("dataset_dtypes", {})
            
            # Evaluate type compatibility for each mapping
            compatible_count = 0
            incompatible_fields = []
            total_inspected = 0
            
            for mapping in mapping_plan:
                if not isinstance(mapping, dict):
                    continue
                
                target_col = mapping.get("target_column")
                source_col = mapping.get("source_column")
                
                if not target_col or target_col not in self.expected_types:
                    continue
                
                total_inspected += 1
                expected_type = self.expected_types[target_col]
                
                # Get observed type (default to string if unknown)
                observed_type = dataset_dtypes.get(source_col, "string")
                
                if self._is_type_compatible(observed_type, expected_type):
                    compatible_count += 1
                else:
                    incompatible_fields.append({
                        "field": target_col,
                        "expected": expected_type,
                        "observed": observed_type,
                    })
            
            # Calculate score
            if total_inspected == 0:
                self.score = 1.0  # No fields to check means no incompatibilities
            else:
                self.score = compatible_count / total_inspected
            
            # Store breakdown
            self.score_breakdown = {
                "inspected_fields": total_inspected,
                "compatible_fields": compatible_count,
                "incompatible_fields": incompatible_fields,
                "compatibility_ratio": self.score,
            }
            
            # Determine success
            self.success = self.score >= self.threshold
            
            # Generate reason
            if self.success:
                self.reason = f"Type compatibility of {self.score:.2%} meets threshold {self.threshold:.2%}"
            else:
                self.reason = (
                    f"Type compatibility of {self.score:.2%} below threshold {self.threshold:.2%}. "
                    f"Found {len(incompatible_fields)} incompatible field(s)"
                )
            
            return self.score
            
        except Exception as e:
            self.error = str(e)
            self.score = 0.0
            self.success = False
            raise
    
    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async version of measure."""
        return self.measure(test_case)
    
    def is_successful(self) -> bool:
        """Check if metric evaluation was successful."""
        if self.error is not None:
            self.success = False
        return self.success if self.success is not None else False
    
    @property
    def __name__(self):
        return "Type Compatibility"


class SemanticSimilarityMetric(BaseMetric):
    """
    Metric to evaluate semantic similarity between source and target column names.
    
    Uses token-based similarity to check if column mappings make semantic sense.
    Score = average similarity across all mappings
    """
    
    def __init__(
        self,
        minimum_score: float = 0.5,
        threshold: float = 0.5,
        strict_mode: bool = False,
        async_mode: bool = False,
    ):
        self.minimum_score = minimum_score
        self.threshold = threshold
        self.strict_mode = strict_mode
        self.async_mode = async_mode
        self.score: Optional[float] = None
        self.success: Optional[bool] = None
        self.score_breakdown: Dict[str, Any] = {}
        self.reason: Optional[str] = None
        self.error: Optional[str] = None
    
    def _calculate_token_similarity(self, source: str, target: str) -> float:
        """Calculate token-based similarity between two column names."""
        # Normalize strings
        source_norm = source.lower().replace("_", " ").replace("-", " ")
        target_norm = target.lower().replace("_", " ").replace("-", " ")
        
        # Split into tokens
        source_tokens = set(source_norm.split())
        target_tokens = set(target_norm.split())
        
        # Handle empty sets
        if not source_tokens or not target_tokens:
            return 1.0 if source_norm == target_norm else 0.0
        
        # Jaccard similarity
        intersection = len(source_tokens & target_tokens)
        union = len(source_tokens | target_tokens)
        
        if union == 0:
            return 1.0 if source == target else 0.0
        
        jaccard = intersection / union
        
        # Bonus for substring matches
        substring_bonus = 0.0
        if source_norm in target_norm or target_norm in source_norm:
            substring_bonus = 0.2
        
        # Common abbreviation patterns
        abbrev_patterns = {
            "id": "identifier",
            "qty": "quantity",
            "amt": "amount",
            "num": "number",
            "prod": "product",
            "desc": "description",
            "cat": "category",
            "pct": "percent",
            "avg": "average",
        }
        
        abbrev_bonus = 0.0
        for abbrev, full in abbrev_patterns.items():
            if (abbrev in source_tokens and full in target_tokens) or \
               (full in source_tokens and abbrev in target_tokens):
                abbrev_bonus = 0.15
                break
        
        return min(1.0, jaccard + substring_bonus + abbrev_bonus)
    
    def measure(self, test_case: LLMTestCase) -> float:
        """Calculate semantic similarity score."""
        try:
            # Extract mappings
            metadata = test_case.additional_metadata or {}
            mapping_plan = metadata.get("mapping_plan", [])
            
            # Calculate similarity for each mapping
            similarity_scores = []
            mapping_details = []
            
            for mapping in mapping_plan:
                if not isinstance(mapping, dict):
                    continue
                
                source_col = mapping.get("source_column", "")
                target_col = mapping.get("target_column", "")
                
                if not source_col or not target_col:
                    continue
                
                similarity = self._calculate_token_similarity(source_col, target_col)
                similarity_scores.append(similarity)
                
                mapping_details.append({
                    "source": source_col,
                    "target": target_col,
                    "similarity": round(similarity, 3),
                })
            
            # Calculate average score
            if len(similarity_scores) == 0:
                self.score = 0.0
            else:
                self.score = sum(similarity_scores) / len(similarity_scores)
            
            # Find low-scoring mappings
            low_scoring = [m for m in mapping_details if m["similarity"] < self.minimum_score]
            
            # Store breakdown
            self.score_breakdown = {
                "average_similarity": self.score,
                "total_mappings": len(mapping_details),
                "low_scoring_mappings": low_scoring,
                "all_mappings": mapping_details[:10],  # Limit to first 10 for brevity
            }
            
            # Determine success
            self.success = self.score >= self.threshold
            
            # Generate reason
            if self.success:
                self.reason = f"Semantic similarity of {self.score:.2%} meets threshold {self.threshold:.2%}"
            else:
                self.reason = (
                    f"Semantic similarity of {self.score:.2%} below threshold {self.threshold:.2%}. "
                    f"Found {len(low_scoring)} low-scoring mapping(s)"
                )
            
            return self.score
            
        except Exception as e:
            self.error = str(e)
            self.score = 0.0
            self.success = False
            raise
    
    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async version of measure."""
        return self.measure(test_case)
    
    def is_successful(self) -> bool:
        """Check if metric evaluation was successful."""
        if self.error is not None:
            self.success = False
        return self.success if self.success is not None else False
    
    @property
    def __name__(self):
        return "Semantic Similarity"

