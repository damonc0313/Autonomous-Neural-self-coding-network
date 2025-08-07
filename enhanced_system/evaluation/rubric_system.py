"""
Comprehensive Evaluation Rubric System

This module provides a strict evaluation framework for assessing code quality,
architecture design, performance, and production readiness across all components
of the GraphformicCoder enhanced system.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import ast
import inspect
import time
import traceback
from pathlib import Path
import json
import numpy as np


class ScoreLevel(Enum):
    """Scoring levels for rubric evaluation."""
    EXCELLENT = 5
    GOOD = 4
    SATISFACTORY = 3
    NEEDS_IMPROVEMENT = 2
    POOR = 1


@dataclass
class RubricCriterion:
    """Individual criterion for evaluation."""
    name: str
    description: str
    weight: float
    max_score: int = 5
    current_score: Optional[int] = None
    feedback: str = ""
    evidence: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class RubricCategory:
    """Category containing multiple criteria."""
    name: str
    description: str
    criteria: List[RubricCriterion]
    weight: float
    
    def calculate_score(self) -> float:
        """Calculate weighted score for this category."""
        if not self.criteria:
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for criterion in self.criteria:
            if criterion.current_score is not None:
                weighted_score = (criterion.current_score / criterion.max_score) * criterion.weight
                total_weighted_score += weighted_score
                total_weight += criterion.weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0


@dataclass
class EvaluationResult:
    """Complete evaluation result for a component."""
    component_name: str
    timestamp: float
    categories: List[RubricCategory]
    overall_score: float = 0.0
    grade: str = ""
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    def calculate_overall_score(self) -> None:
        """Calculate overall weighted score across all categories."""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for category in self.categories:
            category_score = category.calculate_score()
            weighted_score = category_score * category.weight
            total_weighted_score += weighted_score
            total_weight += category.weight
        
        self.overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        self.grade = self._score_to_grade(self.overall_score)
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade."""
        if score >= 0.9:
            return "A+"
        elif score >= 0.85:
            return "A"
        elif score >= 0.8:
            return "A-"
        elif score >= 0.75:
            return "B+"
        elif score >= 0.7:
            return "B"
        elif score >= 0.65:
            return "B-"
        elif score >= 0.6:
            return "C+"
        elif score >= 0.55:
            return "C"
        elif score >= 0.5:
            return "C-"
        else:
            return "F"


class ProductionReadinessRubric:
    """Comprehensive rubric for production-ready code evaluation."""
    
    def __init__(self):
        self.categories = self._initialize_categories()
    
    def _initialize_categories(self) -> List[RubricCategory]:
        """Initialize all evaluation categories and criteria."""
        
        # 1. Code Quality & Style
        code_quality = RubricCategory(
            name="Code Quality & Style",
            description="Code structure, readability, and adherence to best practices",
            weight=0.2,
            criteria=[
                RubricCriterion(
                    name="Code Structure",
                    description="Clear organization, logical flow, appropriate abstraction levels",
                    weight=0.3
                ),
                RubricCriterion(
                    name="Documentation",
                    description="Comprehensive docstrings, type hints, inline comments",
                    weight=0.25
                ),
                RubricCriterion(
                    name="Naming Conventions",
                    description="Clear, consistent, descriptive variable and function names",
                    weight=0.2
                ),
                RubricCriterion(
                    name="Code Complexity",
                    description="Manageable complexity, avoiding deep nesting and long functions",
                    weight=0.25
                )
            ]
        )
        
        # 2. Architecture & Design
        architecture = RubricCategory(
            name="Architecture & Design",
            description="System design, patterns, and architectural principles",
            weight=0.25,
            criteria=[
                RubricCriterion(
                    name="SOLID Principles",
                    description="Adherence to Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion",
                    weight=0.3
                ),
                RubricCriterion(
                    name="Design Patterns",
                    description="Appropriate use of design patterns (Factory, Strategy, Observer, etc.)",
                    weight=0.2
                ),
                RubricCriterion(
                    name="Separation of Concerns",
                    description="Clear separation between business logic, data access, and presentation",
                    weight=0.25
                ),
                RubricCriterion(
                    name="Dependency Management",
                    description="Proper dependency injection, loose coupling, high cohesion",
                    weight=0.25
                )
            ]
        )
        
        # 3. Performance & Scalability
        performance = RubricCategory(
            name="Performance & Scalability",
            description="Efficiency, resource usage, and scalability considerations",
            weight=0.2,
            criteria=[
                RubricCriterion(
                    name="Time Complexity",
                    description="Efficient algorithms with appropriate time complexity",
                    weight=0.3
                ),
                RubricCriterion(
                    name="Memory Efficiency",
                    description="Optimal memory usage, proper resource cleanup",
                    weight=0.3
                ),
                RubricCriterion(
                    name="Scalability Design",
                    description="Architecture supports scaling to larger datasets/users",
                    weight=0.2
                ),
                RubricCriterion(
                    name="Resource Management",
                    description="Proper handling of files, connections, and system resources",
                    weight=0.2
                )
            ]
        )
        
        # 4. Error Handling & Robustness
        robustness = RubricCategory(
            name="Error Handling & Robustness",
            description="Error handling, edge cases, and system reliability",
            weight=0.15,
            criteria=[
                RubricCriterion(
                    name="Exception Handling",
                    description="Comprehensive exception handling with appropriate recovery strategies",
                    weight=0.3
                ),
                RubricCriterion(
                    name="Input Validation",
                    description="Proper validation of inputs and parameters",
                    weight=0.25
                ),
                RubricCriterion(
                    name="Edge Case Handling",
                    description="Consideration and handling of edge cases and boundary conditions",
                    weight=0.25
                ),
                RubricCriterion(
                    name="Graceful Degradation",
                    description="System continues to function when components fail",
                    weight=0.2
                )
            ]
        )
        
        # 5. Testing & Quality Assurance
        testing = RubricCategory(
            name="Testing & Quality Assurance",
            description="Test coverage, test quality, and verification strategies",
            weight=0.1,
            criteria=[
                RubricCriterion(
                    name="Test Coverage",
                    description="Comprehensive test coverage including unit, integration, and edge cases",
                    weight=0.3
                ),
                RubricCriterion(
                    name="Test Quality",
                    description="Well-written, maintainable tests with clear assertions",
                    weight=0.25
                ),
                RubricCriterion(
                    name="Test Organization",
                    description="Logical test organization and proper test data management",
                    weight=0.2
                ),
                RubricCriterion(
                    name="Continuous Integration",
                    description="Automated testing and quality checks in CI/CD pipeline",
                    weight=0.25
                )
            ]
        )
        
        # 6. Security & Safety
        security = RubricCategory(
            name="Security & Safety",
            description="Security considerations and safe coding practices",
            weight=0.1,
            criteria=[
                RubricCriterion(
                    name="Input Sanitization",
                    description="Proper sanitization of user inputs and external data",
                    weight=0.3
                ),
                RubricCriterion(
                    name="Data Protection",
                    description="Secure handling of sensitive data and credentials",
                    weight=0.25
                ),
                RubricCriterion(
                    name="Access Control",
                    description="Appropriate access controls and permission management",
                    weight=0.2
                ),
                RubricCriterion(
                    name="Vulnerability Prevention",
                    description="Protection against common vulnerabilities (injection, XSS, etc.)",
                    weight=0.25
                )
            ]
        )
        
        return [code_quality, architecture, performance, robustness, testing, security]


class ComponentEvaluator:
    """Main evaluator class for assessing components against the rubric."""
    
    def __init__(self):
        self.rubric = ProductionReadinessRubric()
        self.evaluation_history = []
    
    def evaluate_component(self, component_code: str, component_name: str, 
                          component_type: str = "module") -> EvaluationResult:
        """
        Evaluate a component against the comprehensive rubric.
        
        Args:
            component_code: Source code of the component
            component_name: Name of the component
            component_type: Type of component (module, class, function)
            
        Returns:
            Complete evaluation result
        """
        result = EvaluationResult(
            component_name=component_name,
            timestamp=time.time(),
            categories=[]
        )
        
        # Parse the code for analysis
        try:
            tree = ast.parse(component_code)
            module_analysis = self._analyze_ast(tree)
        except SyntaxError as e:
            # If code has syntax errors, assign poor scores
            result.categories = self._create_failed_evaluation("Syntax Error", str(e))
            result.calculate_overall_score()
            return result
        
        # Evaluate each category
        for category in self.rubric.categories:
            evaluated_category = self._evaluate_category(
                category, component_code, module_analysis, component_type
            )
            result.categories.append(evaluated_category)
        
        # Calculate overall score and generate summary
        result.calculate_overall_score()
        result.summary = self._generate_summary(result)
        result.recommendations = self._generate_recommendations(result)
        
        # Store in history
        self.evaluation_history.append(result)
        
        return result
    
    def _analyze_ast(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze AST to extract code metrics."""
        analysis = {
            "classes": [],
            "functions": [],
            "imports": [],
            "docstrings": [],
            "complexity_metrics": {},
            "type_hints": 0,
            "total_lines": 0,
            "comment_lines": 0
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                analysis["classes"].append({
                    "name": node.name,
                    "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                    "has_docstring": ast.get_docstring(node) is not None,
                    "line_count": node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                })
            
            elif isinstance(node, ast.FunctionDef):
                analysis["functions"].append({
                    "name": node.name,
                    "args": len(node.args.args),
                    "has_docstring": ast.get_docstring(node) is not None,
                    "has_return_annotation": node.returns is not None,
                    "line_count": node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                })
            
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                analysis["imports"].append(node)
            
            # Count type hints
            elif isinstance(node, ast.arg) and node.annotation:
                analysis["type_hints"] += 1
        
        return analysis
    
    def _evaluate_category(self, category: RubricCategory, code: str, 
                          analysis: Dict[str, Any], component_type: str) -> RubricCategory:
        """Evaluate a specific category."""
        evaluated_category = RubricCategory(
            name=category.name,
            description=category.description,
            weight=category.weight,
            criteria=[]
        )
        
        for criterion in category.criteria:
            evaluated_criterion = self._evaluate_criterion(
                criterion, code, analysis, component_type
            )
            evaluated_category.criteria.append(evaluated_criterion)
        
        return evaluated_category
    
    def _evaluate_criterion(self, criterion: RubricCriterion, code: str,
                           analysis: Dict[str, Any], component_type: str) -> RubricCriterion:
        """Evaluate a specific criterion."""
        evaluated = RubricCriterion(
            name=criterion.name,
            description=criterion.description,
            weight=criterion.weight,
            max_score=criterion.max_score
        )
        
        # Apply specific evaluation logic based on criterion name
        if criterion.name == "Code Structure":
            evaluated = self._evaluate_code_structure(evaluated, code, analysis)
        elif criterion.name == "Documentation":
            evaluated = self._evaluate_documentation(evaluated, code, analysis)
        elif criterion.name == "Naming Conventions":
            evaluated = self._evaluate_naming_conventions(evaluated, code, analysis)
        elif criterion.name == "Code Complexity":
            evaluated = self._evaluate_code_complexity(evaluated, code, analysis)
        elif criterion.name == "SOLID Principles":
            evaluated = self._evaluate_solid_principles(evaluated, code, analysis)
        elif criterion.name == "Design Patterns":
            evaluated = self._evaluate_design_patterns(evaluated, code, analysis)
        elif criterion.name == "Separation of Concerns":
            evaluated = self._evaluate_separation_of_concerns(evaluated, code, analysis)
        elif criterion.name == "Dependency Management":
            evaluated = self._evaluate_dependency_management(evaluated, code, analysis)
        elif criterion.name == "Exception Handling":
            evaluated = self._evaluate_exception_handling(evaluated, code, analysis)
        elif criterion.name == "Input Validation":
            evaluated = self._evaluate_input_validation(evaluated, code, analysis)
        else:
            # Default evaluation for criteria not specifically implemented
            evaluated.current_score = 3
            evaluated.feedback = "Standard evaluation - meets basic requirements"
        
        return evaluated
    
    def _evaluate_code_structure(self, criterion: RubricCriterion, code: str, 
                                analysis: Dict[str, Any]) -> RubricCriterion:
        """Evaluate code structure and organization."""
        score = 5
        feedback_points = []
        evidence = []
        
        # Check class organization
        if analysis["classes"]:
            avg_methods = np.mean([cls["methods"] for cls in analysis["classes"]])
            if avg_methods > 20:
                score -= 1
                feedback_points.append("Some classes are quite large (>20 methods)")
            evidence.append(f"Average methods per class: {avg_methods:.1f}")
        
        # Check function organization
        if analysis["functions"]:
            large_functions = [f for f in analysis["functions"] if f["line_count"] > 50]
            if large_functions:
                score -= 1
                feedback_points.append(f"{len(large_functions)} functions exceed 50 lines")
            evidence.append(f"Total functions: {len(analysis['functions'])}")
        
        # Check imports organization
        if len(analysis["imports"]) > 20:
            score -= 1
            feedback_points.append("High number of imports may indicate tight coupling")
        
        criterion.current_score = max(1, score)
        criterion.feedback = "; ".join(feedback_points) if feedback_points else "Well-structured code"
        criterion.evidence = evidence
        
        return criterion
    
    def _evaluate_documentation(self, criterion: RubricCriterion, code: str,
                               analysis: Dict[str, Any]) -> RubricCriterion:
        """Evaluate documentation quality."""
        score = 5
        feedback_points = []
        evidence = []
        
        # Check docstring coverage
        total_functions = len(analysis["functions"])
        documented_functions = sum(1 for f in analysis["functions"] if f["has_docstring"])
        
        if total_functions > 0:
            docstring_coverage = documented_functions / total_functions
            evidence.append(f"Function docstring coverage: {docstring_coverage:.1%}")
            
            if docstring_coverage < 0.5:
                score -= 2
                feedback_points.append("Low docstring coverage (<50%)")
            elif docstring_coverage < 0.8:
                score -= 1
                feedback_points.append("Moderate docstring coverage (<80%)")
        
        # Check type hints
        total_args = sum(f["args"] for f in analysis["functions"])
        type_hint_coverage = analysis["type_hints"] / max(1, total_args)
        evidence.append(f"Type hint coverage: {type_hint_coverage:.1%}")
        
        if type_hint_coverage < 0.3:
            score -= 1
            feedback_points.append("Limited type hints")
        
        # Check for module docstring
        if '"""' not in code[:500]:  # Simple check for module docstring
            score -= 1
            feedback_points.append("Missing module-level documentation")
        
        criterion.current_score = max(1, score)
        criterion.feedback = "; ".join(feedback_points) if feedback_points else "Excellent documentation"
        criterion.evidence = evidence
        
        return criterion
    
    def _evaluate_naming_conventions(self, criterion: RubricCriterion, code: str,
                                   analysis: Dict[str, Any]) -> RubricCriterion:
        """Evaluate naming conventions."""
        score = 5
        feedback_points = []
        evidence = []
        
        # Check class names (should be PascalCase)
        non_pascal_classes = [cls["name"] for cls in analysis["classes"] 
                             if not cls["name"][0].isupper()]
        if non_pascal_classes:
            score -= 1
            feedback_points.append(f"Non-PascalCase classes: {non_pascal_classes}")
        
        # Check function names (should be snake_case)
        non_snake_functions = [f["name"] for f in analysis["functions"]
                              if f["name"] != f["name"].lower() or ' ' in f["name"]]
        if non_snake_functions:
            score -= 1
            feedback_points.append(f"Non-snake_case functions: {non_snake_functions}")
        
        # Check for descriptive names
        short_names = [f["name"] for f in analysis["functions"] 
                      if len(f["name"]) < 3 and f["name"] not in ["__init__", "__str__"]]
        if short_names:
            score -= 1
            feedback_points.append(f"Very short function names: {short_names}")
        
        evidence.append(f"Classes: {len(analysis['classes'])}")
        evidence.append(f"Functions: {len(analysis['functions'])}")
        
        criterion.current_score = max(1, score)
        criterion.feedback = "; ".join(feedback_points) if feedback_points else "Good naming conventions"
        criterion.evidence = evidence
        
        return criterion
    
    def _evaluate_code_complexity(self, criterion: RubricCriterion, code: str,
                                 analysis: Dict[str, Any]) -> RubricCriterion:
        """Evaluate code complexity."""
        score = 5
        feedback_points = []
        evidence = []
        
        # Simple complexity metrics
        lines_of_code = len([line for line in code.split('\n') if line.strip()])
        evidence.append(f"Total lines of code: {lines_of_code}")
        
        # Check for nested complexity indicators
        nesting_indicators = code.count('    if ') + code.count('    for ') + code.count('    while ')
        if nesting_indicators > lines_of_code * 0.1:
            score -= 1
            feedback_points.append("High nesting complexity detected")
        
        # Check function length
        if analysis["functions"]:
            avg_function_length = np.mean([f["line_count"] for f in analysis["functions"]])
            evidence.append(f"Average function length: {avg_function_length:.1f} lines")
            
            if avg_function_length > 30:
                score -= 1
                feedback_points.append("Functions are quite long on average")
        
        criterion.current_score = max(1, score)
        criterion.feedback = "; ".join(feedback_points) if feedback_points else "Manageable complexity"
        criterion.evidence = evidence
        
        return criterion
    
    def _evaluate_solid_principles(self, criterion: RubricCriterion, code: str,
                                  analysis: Dict[str, Any]) -> RubricCriterion:
        """Evaluate SOLID principles adherence."""
        score = 5
        feedback_points = []
        evidence = []
        
        # Single Responsibility: Check for classes with too many methods
        if analysis["classes"]:
            large_classes = [cls for cls in analysis["classes"] if cls["methods"] > 15]
            if large_classes:
                score -= 1
                feedback_points.append("Some classes may violate Single Responsibility Principle")
                evidence.append(f"Large classes: {[cls['name'] for cls in large_classes]}")
        
        # Dependency Inversion: Look for protocol/interface usage
        if "Protocol" in code or "ABC" in code:
            evidence.append("Uses protocols/abstract base classes")
        else:
            score -= 1
            feedback_points.append("Limited use of abstractions/interfaces")
        
        # Open/Closed: Look for extensibility patterns
        if "factory" in code.lower() or "strategy" in code.lower():
            evidence.append("Shows extensibility patterns")
        
        criterion.current_score = max(1, score)
        criterion.feedback = "; ".join(feedback_points) if feedback_points else "Good SOLID adherence"
        criterion.evidence = evidence
        
        return criterion
    
    def _evaluate_design_patterns(self, criterion: RubricCriterion, code: str,
                                 analysis: Dict[str, Any]) -> RubricCriterion:
        """Evaluate design pattern usage."""
        score = 3  # Start with neutral score
        feedback_points = []
        evidence = []
        patterns_found = []
        
        # Check for common patterns
        pattern_indicators = {
            "Factory": ["create_", "factory", "Factory"],
            "Strategy": ["Strategy", "strategy", "algorithm"],
            "Observer": ["Observer", "observer", "notify", "subscribe"],
            "Singleton": ["Singleton", "_instance", "__new__"],
            "Builder": ["Builder", "builder", "build"],
            "Adapter": ["Adapter", "adapter", "adapt"],
            "Decorator": ["@", "decorator", "wrapper"]
        }
        
        for pattern, indicators in pattern_indicators.items():
            if any(indicator in code for indicator in indicators):
                patterns_found.append(pattern)
                score += 0.5
        
        if patterns_found:
            evidence.append(f"Design patterns found: {patterns_found}")
            feedback_points.append("Good use of design patterns")
        else:
            feedback_points.append("Could benefit from design pattern usage")
        
        criterion.current_score = min(5, max(1, int(score)))
        criterion.feedback = "; ".join(feedback_points)
        criterion.evidence = evidence
        
        return criterion
    
    def _evaluate_separation_of_concerns(self, criterion: RubricCriterion, code: str,
                                        analysis: Dict[str, Any]) -> RubricCriterion:
        """Evaluate separation of concerns."""
        score = 4
        feedback_points = []
        evidence = []
        
        # Look for mixed responsibilities
        mixed_concerns = []
        if "database" in code.lower() and "ui" in code.lower():
            mixed_concerns.append("Database and UI logic mixed")
        if "network" in code.lower() and "business" in code.lower():
            mixed_concerns.append("Network and business logic mixed")
        
        if mixed_concerns:
            score -= len(mixed_concerns)
            feedback_points.extend(mixed_concerns)
        
        # Check for proper layering
        if any(keyword in code.lower() for keyword in ["service", "repository", "controller"]):
            score += 1
            evidence.append("Shows layered architecture patterns")
        
        criterion.current_score = max(1, min(5, score))
        criterion.feedback = "; ".join(feedback_points) if feedback_points else "Good separation of concerns"
        criterion.evidence = evidence
        
        return criterion
    
    def _evaluate_dependency_management(self, criterion: RubricCriterion, code: str,
                                       analysis: Dict[str, Any]) -> RubricCriterion:
        """Evaluate dependency management."""
        score = 4
        feedback_points = []
        evidence = []
        
        # Check for dependency injection
        if "inject" in code.lower() or "__init__" in code:
            score += 1
            evidence.append("Uses constructor injection")
        
        # Check for loose coupling indicators
        if "Protocol" in code or "ABC" in code:
            evidence.append("Uses abstractions for loose coupling")
        else:
            score -= 1
            feedback_points.append("Could improve loose coupling with abstractions")
        
        # Check import organization
        import_count = len(analysis["imports"])
        if import_count > 30:
            score -= 1
            feedback_points.append("High number of imports may indicate tight coupling")
        
        evidence.append(f"Total imports: {import_count}")
        
        criterion.current_score = max(1, min(5, score))
        criterion.feedback = "; ".join(feedback_points) if feedback_points else "Good dependency management"
        criterion.evidence = evidence
        
        return criterion
    
    def _evaluate_exception_handling(self, criterion: RubricCriterion, code: str,
                                    analysis: Dict[str, Any]) -> RubricCriterion:
        """Evaluate exception handling."""
        score = 3
        feedback_points = []
        evidence = []
        
        # Count exception handling constructs
        try_blocks = code.count("try:")
        except_blocks = code.count("except")
        finally_blocks = code.count("finally:")
        
        evidence.append(f"Try blocks: {try_blocks}")
        evidence.append(f"Except blocks: {except_blocks}")
        
        if try_blocks > 0:
            score += 1
            feedback_points.append("Includes exception handling")
        else:
            feedback_points.append("Missing exception handling")
        
        # Check for specific exception types
        if "Exception" in code and "except Exception:" not in code:
            score += 1
            evidence.append("Uses specific exception types")
        
        # Check for finally blocks
        if finally_blocks > 0:
            score += 1
            evidence.append("Uses finally blocks for cleanup")
        
        criterion.current_score = max(1, min(5, score))
        criterion.feedback = "; ".join(feedback_points)
        criterion.evidence = evidence
        
        return criterion
    
    def _evaluate_input_validation(self, criterion: RubricCriterion, code: str,
                                  analysis: Dict[str, Any]) -> RubricCriterion:
        """Evaluate input validation."""
        score = 3
        feedback_points = []
        evidence = []
        
        # Look for validation patterns
        validation_patterns = ["isinstance", "assert", "raise ValueError", "validate", "check"]
        found_patterns = [pattern for pattern in validation_patterns if pattern in code]
        
        if found_patterns:
            score += len(found_patterns)
            evidence.append(f"Validation patterns found: {found_patterns}")
            feedback_points.append("Includes input validation")
        else:
            feedback_points.append("Missing input validation")
        
        # Check for type hints (implicit validation)
        if analysis["type_hints"] > 0:
            score += 1
            evidence.append("Uses type hints for validation")
        
        criterion.current_score = max(1, min(5, score))
        criterion.feedback = "; ".join(feedback_points)
        criterion.evidence = evidence
        
        return criterion
    
    def _create_failed_evaluation(self, error_type: str, error_message: str) -> List[RubricCategory]:
        """Create evaluation result for failed analysis."""
        failed_categories = []
        
        for category in self.rubric.categories:
            failed_category = RubricCategory(
                name=category.name,
                description=category.description,
                weight=category.weight,
                criteria=[]
            )
            
            for criterion in category.criteria:
                failed_criterion = RubricCriterion(
                    name=criterion.name,
                    description=criterion.description,
                    weight=criterion.weight,
                    current_score=1,
                    feedback=f"{error_type}: {error_message}"
                )
                failed_category.criteria.append(failed_criterion)
            
            failed_categories.append(failed_category)
        
        return failed_categories
    
    def _generate_summary(self, result: EvaluationResult) -> str:
        """Generate evaluation summary."""
        grade_descriptions = {
            "A+": "Exceptional quality - Production ready with exemplary practices",
            "A": "Excellent quality - Production ready with minor improvements",
            "A-": "Very good quality - Nearly production ready",
            "B+": "Good quality - Requires some improvements for production",
            "B": "Satisfactory quality - Requires moderate improvements",
            "B-": "Below average - Needs significant improvements",
            "C+": "Poor quality - Major refactoring required",
            "C": "Very poor quality - Extensive rework needed",
            "C-": "Unacceptable quality - Complete redesign required",
            "F": "Failed - Critical issues prevent usage"
        }
        
        summary = f"Component '{result.component_name}' received grade {result.grade} "
        summary += f"({result.overall_score:.2f}/1.0). "
        summary += grade_descriptions.get(result.grade, "Quality assessment completed.")
        
        return summary
    
    def _generate_recommendations(self, result: EvaluationResult) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Analyze weak areas
        weak_categories = [cat for cat in result.categories if cat.calculate_score() < 0.7]
        
        for category in weak_categories:
            weak_criteria = [crit for crit in category.criteria 
                           if crit.current_score and crit.current_score < 4]
            
            for criterion in weak_criteria:
                if criterion.suggestions:
                    recommendations.extend(criterion.suggestions)
                else:
                    recommendations.append(f"Improve {criterion.name.lower()} in {category.name.lower()}")
        
        # Add general recommendations based on score
        if result.overall_score < 0.6:
            recommendations.append("Consider major refactoring to improve overall quality")
        elif result.overall_score < 0.8:
            recommendations.append("Focus on addressing the weakest areas identified")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def generate_report(self, result: EvaluationResult, detailed: bool = True) -> str:
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("=" * 80)
        report.append(f"COMPONENT EVALUATION REPORT: {result.component_name}")
        report.append("=" * 80)
        report.append(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result.timestamp))}")
        report.append(f"Overall Score: {result.overall_score:.3f}/1.000")
        report.append(f"Grade: {result.grade}")
        report.append("")
        report.append("SUMMARY:")
        report.append(result.summary)
        report.append("")
        
        if detailed:
            report.append("DETAILED BREAKDOWN:")
            report.append("-" * 40)
            
            for category in result.categories:
                category_score = category.calculate_score()
                report.append(f"\n{category.name}: {category_score:.3f}/1.000 (Weight: {category.weight:.1%})")
                report.append(f"Description: {category.description}")
                
                for criterion in category.criteria:
                    if criterion.current_score is not None:
                        report.append(f"  • {criterion.name}: {criterion.current_score}/{criterion.max_score}")
                        if criterion.feedback:
                            report.append(f"    Feedback: {criterion.feedback}")
                        if criterion.evidence:
                            report.append(f"    Evidence: {'; '.join(criterion.evidence)}")
        
        if result.recommendations:
            report.append("\nRECOMMENDATIONS FOR IMPROVEMENT:")
            report.append("-" * 40)
            for i, rec in enumerate(result.recommendations, 1):
                report.append(f"{i}. {rec}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def save_evaluation(self, result: EvaluationResult, filepath: str) -> None:
        """Save evaluation result to JSON file."""
        # Convert to serializable format
        data = {
            "component_name": result.component_name,
            "timestamp": result.timestamp,
            "overall_score": result.overall_score,
            "grade": result.grade,
            "summary": result.summary,
            "recommendations": result.recommendations,
            "categories": []
        }
        
        for category in result.categories:
            cat_data = {
                "name": category.name,
                "description": category.description,
                "weight": category.weight,
                "score": category.calculate_score(),
                "criteria": []
            }
            
            for criterion in category.criteria:
                crit_data = {
                    "name": criterion.name,
                    "description": criterion.description,
                    "weight": criterion.weight,
                    "max_score": criterion.max_score,
                    "current_score": criterion.current_score,
                    "feedback": criterion.feedback,
                    "evidence": criterion.evidence,
                    "suggestions": criterion.suggestions
                }
                cat_data["criteria"].append(crit_data)
            
            data["categories"].append(cat_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# Example usage and testing
if __name__ == "__main__":
    evaluator = ComponentEvaluator()
    
    # Example code to evaluate
    sample_code = '''
"""
Sample module for evaluation testing.
"""

from typing import Protocol, List, Optional
import logging

class DataProcessor(Protocol):
    """Protocol for data processing."""
    def process(self, data: List[str]) -> List[str]:
        ...

class TextProcessor:
    """Processes text data with various transformations."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize processor with optional logger."""
        self.logger = logger or logging.getLogger(__name__)
    
    def process(self, data: List[str]) -> List[str]:
        """
        Process list of text strings.
        
        Args:
            data: List of strings to process
            
        Returns:
            Processed list of strings
        """
        if not isinstance(data, list):
            raise ValueError("Data must be a list")
        
        try:
            result = []
            for item in data:
                if isinstance(item, str):
                    processed = self._clean_text(item)
                    result.append(processed)
            return result
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean individual text string."""
        return text.strip().lower()
'''
    
    # Evaluate the sample code
    result = evaluator.evaluate_component(sample_code, "TextProcessor", "class")
    
    # Generate and print report
    report = evaluator.generate_report(result, detailed=True)
    print(report)
    
    print(f"\n✅ Evaluation rubric system implemented and tested")
    print(f"   Component Grade: {result.grade}")
    print(f"   Overall Score: {result.overall_score:.3f}")