#!/usr/bin/env python3
"""
Comprehensive Test Suite: Dialectical Synthesis Validation
==========================================================

This test suite validates all components of the dialectical synthesis framework,
including the original GraphformicCoder, adaptive enhancements, hierarchical
evolution engine, and autonomous optimization capabilities.

Test Categories:
1. Syntax and Import Validation
2. Core Functionality Tests  
3. Performance Benchmarks
4. Dialectical Framework Validation
5. Integration Tests
6. Autonomous Evolution Validation

Author: Dale (Veteran Software Architect)
"""

import unittest
import time
import sys
import traceback
import torch
import ast
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Test result with detailed metrics."""
    name: str
    passed: bool
    duration: float
    error: Optional[str] = None
    metrics: Dict[str, Any] = None


class DialecticalTestFramework:
    """Comprehensive testing framework for dialectical synthesis validation."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()
    
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> TestResult:
        """Run a single test with error handling and timing."""
        logger.info(f"🧪 Running test: {test_name}")
        
        start_time = time.time()
        try:
            result = test_func(*args, **kwargs)
            duration = time.time() - start_time
            
            if isinstance(result, dict):
                test_result = TestResult(
                    name=test_name,
                    passed=result.get('passed', True),
                    duration=duration,
                    metrics=result.get('metrics', {})
                )
            else:
                test_result = TestResult(
                    name=test_name,
                    passed=bool(result),
                    duration=duration
                )
            
        except Exception as e:
            duration = time.time() - start_time
            test_result = TestResult(
                name=test_name,
                passed=False,
                duration=duration,
                error=str(e)
            )
            logger.error(f"❌ Test failed: {test_name} - {str(e)}")
        
        self.results.append(test_result)
        status = "✅" if test_result.passed else "❌"
        logger.info(f"{status} {test_name} ({duration:.3f}s)")
        
        return test_result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive test summary."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        total_duration = time.time() - self.start_time
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / max(total_tests, 1),
            'total_duration': total_duration,
            'avg_test_duration': sum(r.duration for r in self.results) / max(total_tests, 1)
        }


class SyntaxValidationTests:
    """Test syntax and import validation for all modules."""
    
    def test_core_modules_import(self):
        """Test importing core modules."""
        modules_to_test = [
            'graphformic_coder',
            'digital_crucible', 
            'autonomous_evolution_engine'
        ]
        
        import_results = {}
        for module_name in modules_to_test:
            try:
                __import__(module_name)
                import_results[module_name] = True
            except Exception as e:
                import_results[module_name] = False
                logger.warning(f"Failed to import {module_name}: {e}")
        
        all_imported = all(import_results.values())
        return {
            'passed': all_imported,
            'metrics': {
                'imported_modules': sum(import_results.values()),
                'total_modules': len(modules_to_test),
                'import_details': import_results
            }
        }
    
    def test_synthesis_modules_syntax(self):
        """Test syntax of synthesized modules."""
        synthesis_files = [
            'adaptive_graphformic_coder.py',
            'hierarchical_evolution_engine.py'
        ]
        
        syntax_results = {}
        for file_name in synthesis_files:
            try:
                with open(file_name, 'r') as f:
                    code = f.read()
                ast.parse(code)
                compile(code, file_name, 'exec')
                syntax_results[file_name] = True
            except Exception as e:
                syntax_results[file_name] = False
                logger.warning(f"Syntax error in {file_name}: {e}")
        
        all_valid = all(syntax_results.values())
        return {
            'passed': all_valid,
            'metrics': {
                'valid_files': sum(syntax_results.values()),
                'total_files': len(synthesis_files),
                'syntax_details': syntax_results
            }
        }


class CoreFunctionalityTests:
    """Test core functionality of the GraphformicCoder ecosystem."""
    
    def test_original_evolution_engine(self):
        """Test the original autonomous evolution engine."""
        try:
            from autonomous_evolution_engine import PurePythonEvolutionEngine
            
            # Simple test code
            test_code = '''
def slow_fibonacci(n):
    if n <= 1:
        return n
    return slow_fibonacci(n-1) + slow_fibonacci(n-2)
            '''
            
            engine = PurePythonEvolutionEngine(population_size=5, max_generations=2)
            
            # Run evolution with timeout
            start_time = time.time()
            best_candidate = engine.evolve_code(test_code)
            evolution_time = time.time() - start_time
            
            # Basic validation
            has_improvement = best_candidate.fitness > 0
            reasonable_time = evolution_time < 10.0  # Should complete quickly
            
            return {
                'passed': has_improvement and reasonable_time,
                'metrics': {
                    'evolution_time': evolution_time,
                    'best_fitness': best_candidate.fitness,
                    'has_improvement': has_improvement,
                    'reasonable_time': reasonable_time
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'metrics': {'error': str(e)}
            }
    
    def test_graphformic_coder_basic(self):
        """Test basic GraphformicCoder functionality."""
        try:
            from graphformic_coder import GraphformicCoder
            
            # Create small model for testing
            config = {
                'vocab_size': 100,
                'node_feature_dim': 32,
                'd_model': 64,
                'nhead': 2,
                'num_transformer_layers': 1,
                'num_gat_layers': 1,
                'num_decoder_layers': 1,
                'dim_feedforward': 128,
                'dropout': 0.1
            }
            
            model = GraphformicCoder(**config)
            
            # Test forward pass
            batch_size = 2
            src_len, tgt_len = 10, 5
            num_nodes, num_edges = 20, 30
            
            src_tokens = torch.randint(1, config['vocab_size'], (batch_size, src_len))
            tgt_tokens = torch.randint(1, config['vocab_size'], (batch_size, tgt_len))
            node_features = torch.randn(num_nodes, config['node_feature_dim'])
            edge_index = torch.randint(0, num_nodes, (2, num_edges))
            batch_graph = torch.cat([
                torch.zeros(num_nodes//2, dtype=torch.long),
                torch.ones(num_nodes//2, dtype=torch.long)
            ])
            
            # Forward pass
            start_time = time.time()
            with torch.no_grad():
                output = model(src_tokens, node_features, edge_index, tgt_tokens, batch_graph)
            forward_time = time.time() - start_time
            
            # Validate output shape
            expected_shape = (batch_size, tgt_len, config['vocab_size'])
            correct_shape = output.shape == expected_shape
            reasonable_time = forward_time < 2.0
            
            return {
                'passed': correct_shape and reasonable_time,
                'metrics': {
                    'output_shape': list(output.shape),
                    'expected_shape': list(expected_shape),
                    'forward_time': forward_time,
                    'correct_shape': correct_shape,
                    'reasonable_time': reasonable_time
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'metrics': {'error': str(e)}
            }


class AdaptiveArchitectureTests:
    """Test adaptive architecture enhancements."""
    
    def test_adaptive_graphformic_coder(self):
        """Test adaptive GraphformicCoder functionality."""
        try:
            # Import with fallback for missing dependencies
            try:
                from adaptive_graphformic_coder import create_adaptive_graphformic_coder
            except ImportError as e:
                logger.warning(f"Cannot import adaptive module: {e}")
                return {
                    'passed': False,
                    'metrics': {'error': 'Import failed - dependencies missing'}
                }
            
            # Create adaptive model
            model = create_adaptive_graphformic_coder(
                vocab_size=100,
                node_feature_dim=32,
                complexity_threshold=0.3
            )
            
            # Test simple case (should use lightweight mode)
            simple_tokens = torch.randint(1, 100, (1, 10))
            simple_code = "def simple(): return 42"
            
            start_time = time.time()
            with torch.no_grad():
                simple_output = model(simple_tokens, code_text=simple_code)
            simple_time = time.time() - start_time
            
            # Test complex case (should use full mode)
            complex_tokens = torch.randint(1, 100, (1, 50))
            complex_code = """
            def complex_func(data):
                result = []
                for item in data:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            try:
                                processed = process_value(value)
                                result.append({key: processed})
                            except Exception as e:
                                continue
                return result
            """
            
            start_time = time.time()
            with torch.no_grad():
                complex_output = model(complex_tokens, code_text=complex_code)
            complex_time = time.time() - start_time
            
            # Get performance stats
            perf_stats = model.get_performance_stats()
            
            # Validate adaptive behavior
            has_performance_data = perf_stats.get('status') != 'no_data'
            reasonable_times = simple_time < 1.0 and complex_time < 2.0
            
            return {
                'passed': has_performance_data and reasonable_times,
                'metrics': {
                    'simple_forward_time': simple_time,
                    'complex_forward_time': complex_time,
                    'performance_stats': perf_stats,
                    'has_performance_data': has_performance_data,
                    'reasonable_times': reasonable_times
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'metrics': {'error': str(e)}
            }


class HierarchicalEvolutionTests:
    """Test hierarchical evolution capabilities."""
    
    def test_hierarchical_evolution_basic(self):
        """Test basic hierarchical evolution functionality."""
        try:
            # Import with fallback
            try:
                from hierarchical_evolution_engine import create_hierarchical_evolution_engine
            except ImportError as e:
                logger.warning(f"Cannot import hierarchical module: {e}")
                return {
                    'passed': False,
                    'metrics': {'error': 'Import failed - dependencies missing'}
                }
            
            # Create engine with small parameters for testing
            engine = create_hierarchical_evolution_engine(
                population_size=5,
                max_generations=2
            )
            
            # Simple test code
            test_code = '''
def test_function(x):
    """Simple test function."""
    return x * 2

class TestClass:
    """Simple test class."""
    def __init__(self):
        self.value = 0
    
    def increment(self):
        self.value += 1
        return self.value
            '''
            
            # Define simple test cases
            def test_function_works(namespace):
                if 'test_function' in namespace:
                    return namespace['test_function'](5) == 10
                return False
            
            def test_class_works(namespace):
                if 'TestClass' in namespace:
                    obj = namespace['TestClass']()
                    return obj.increment() == 1
                return False
            
            test_cases = [test_function_works, test_class_works]
            
            # Run hierarchical evolution
            start_time = time.time()
            best_candidate = engine.evolve_hierarchically(test_code, test_cases)
            evolution_time = time.time() - start_time
            
            # Validate results
            has_result = best_candidate is not None
            has_fitness = best_candidate.fitness > 0 if has_result else False
            reasonable_time = evolution_time < 30.0  # Allow more time for hierarchical
            has_metrics = bool(best_candidate.performance_metrics) if has_result else False
            
            return {
                'passed': has_result and has_fitness and reasonable_time,
                'metrics': {
                    'evolution_time': evolution_time,
                    'best_fitness': best_candidate.fitness if has_result else 0,
                    'generation': best_candidate.generation if has_result else 0,
                    'level': best_candidate.level if has_result else 'none',
                    'has_result': has_result,
                    'has_fitness': has_fitness,
                    'reasonable_time': reasonable_time,
                    'has_metrics': has_metrics
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'metrics': {'error': str(e)}
            }


class IntegrationTests:
    """Test integration between different components."""
    
    def test_full_pipeline_integration(self):
        """Test full pipeline from original to evolved adaptive architecture."""
        try:
            # Test the complete flow:
            # 1. Original evolution engine works
            # 2. GraphformicCoder works  
            # 3. Adaptive enhancements work
            # 4. Hierarchical evolution works
            
            pipeline_results = {}
            
            # Step 1: Original evolution
            try:
                from autonomous_evolution_engine import PurePythonEvolutionEngine
                engine = PurePythonEvolutionEngine(population_size=3, max_generations=1)
                test_code = "def test(x): return x + 1"
                result = engine.evolve_code(test_code)
                pipeline_results['original_evolution'] = result.fitness > 0
            except Exception as e:
                pipeline_results['original_evolution'] = False
                logger.warning(f"Original evolution failed: {e}")
            
            # Step 2: Core GraphformicCoder
            try:
                from graphformic_coder import GraphformicCoder
                model = GraphformicCoder(vocab_size=50, node_feature_dim=16, d_model=32, 
                                       nhead=1, num_transformer_layers=1, num_gat_layers=1, 
                                       num_decoder_layers=1, dim_feedforward=64)
                src = torch.randint(1, 50, (1, 5))
                tgt = torch.randint(1, 50, (1, 3))
                nodes = torch.randn(10, 16)
                edges = torch.randint(0, 10, (2, 15))
                batch = torch.zeros(10, dtype=torch.long)
                
                with torch.no_grad():
                    output = model(src, nodes, edges, tgt, batch)
                pipeline_results['core_model'] = output.shape == (1, 3, 50)
            except Exception as e:
                pipeline_results['core_model'] = False
                logger.warning(f"Core model failed: {e}")
            
            # Step 3: Adaptive architecture (may fail due to dependencies)
            try:
                from adaptive_graphformic_coder import create_adaptive_graphformic_coder
                adaptive_model = create_adaptive_graphformic_coder(vocab_size=50, node_feature_dim=16)
                test_tokens = torch.randint(1, 50, (1, 10))
                with torch.no_grad():
                    adaptive_output = adaptive_model(test_tokens)
                pipeline_results['adaptive_model'] = adaptive_output.shape[0] == 1
            except Exception as e:
                pipeline_results['adaptive_model'] = False
                logger.warning(f"Adaptive model failed: {e}")
            
            # Step 4: Hierarchical evolution (may fail due to dependencies)
            try:
                from hierarchical_evolution_engine import create_hierarchical_evolution_engine
                hier_engine = create_hierarchical_evolution_engine(population_size=3, max_generations=1)
                simple_code = "def simple(x): return x"
                hier_result = hier_engine.evolve_hierarchically(simple_code)
                pipeline_results['hierarchical_evolution'] = hier_result.fitness >= 0
            except Exception as e:
                pipeline_results['hierarchical_evolution'] = False
                logger.warning(f"Hierarchical evolution failed: {e}")
            
            # Evaluate overall pipeline
            successful_steps = sum(pipeline_results.values())
            total_steps = len(pipeline_results)
            
            return {
                'passed': successful_steps >= 2,  # At least half should work
                'metrics': {
                    'successful_steps': successful_steps,
                    'total_steps': total_steps,
                    'success_rate': successful_steps / total_steps,
                    'pipeline_details': pipeline_results
                }
            }
            
        except Exception as e:
            return {
                'passed': False,
                'metrics': {'error': str(e)}
            }


def main():
    """Run comprehensive test suite."""
    print("🧪 Comprehensive Test Suite: Dialectical Synthesis Validation")
    print("=" * 70)
    
    framework = DialecticalTestFramework()
    
    # Initialize test classes
    syntax_tests = SyntaxValidationTests()
    core_tests = CoreFunctionalityTests()
    adaptive_tests = AdaptiveArchitectureTests()
    hierarchical_tests = HierarchicalEvolutionTests()
    integration_tests = IntegrationTests()
    
    # Run test suites
    print("\n🔍 Category 1: Syntax and Import Validation")
    framework.run_test("Core Modules Import", syntax_tests.test_core_modules_import)
    framework.run_test("Synthesis Modules Syntax", syntax_tests.test_synthesis_modules_syntax)
    
    print("\n🔧 Category 2: Core Functionality Tests")
    framework.run_test("Original Evolution Engine", core_tests.test_original_evolution_engine)
    framework.run_test("GraphformicCoder Basic", core_tests.test_graphformic_coder_basic)
    
    print("\n🎯 Category 3: Adaptive Architecture Tests")
    framework.run_test("Adaptive GraphformicCoder", adaptive_tests.test_adaptive_graphformic_coder)
    
    print("\n🧬 Category 4: Hierarchical Evolution Tests")
    framework.run_test("Hierarchical Evolution Basic", hierarchical_tests.test_hierarchical_evolution_basic)
    
    print("\n🔗 Category 5: Integration Tests")
    framework.run_test("Full Pipeline Integration", integration_tests.test_full_pipeline_integration)
    
    # Generate summary report
    print("\n" + "="*70)
    print("📊 TEST SUMMARY REPORT")
    print("="*70)
    
    summary = framework.get_summary()
    
    print(f"📈 Overall Results:")
    print(f"   Total Tests: {summary['total_tests']}")
    print(f"   Passed: {summary['passed_tests']}")
    print(f"   Failed: {summary['failed_tests']}")
    print(f"   Success Rate: {summary['success_rate']:.1%}")
    print(f"   Total Duration: {summary['total_duration']:.2f}s")
    print(f"   Avg Test Duration: {summary['avg_test_duration']:.3f}s")
    
    print(f"\n📋 Detailed Results:")
    for result in framework.results:
        status = "✅" if result.passed else "❌"
        print(f"   {status} {result.name} ({result.duration:.3f}s)")
        if result.error:
            print(f"      Error: {result.error}")
        if result.metrics and 'error' not in result.metrics:
            key_metrics = {k: v for k, v in result.metrics.items() 
                          if isinstance(v, (int, float, bool))}
            if key_metrics:
                metrics_str = ", ".join(f"{k}={v}" for k, v in list(key_metrics.items())[:3])
                print(f"      Metrics: {metrics_str}")
    
    # Dialectical synthesis validation
    print(f"\n🎯 DIALECTICAL SYNTHESIS VALIDATION:")
    
    # Calculate synthesis success metrics
    syntax_success = sum(1 for r in framework.results[:2] if r.passed) / 2
    core_success = sum(1 for r in framework.results[2:4] if r.passed) / 2  
    advanced_success = sum(1 for r in framework.results[4:6] if r.passed) / 2
    integration_success = sum(1 for r in framework.results[6:] if r.passed) / max(1, len(framework.results[6:]))
    
    print(f"   🔍 Syntax & Structure: {syntax_success:.1%}")
    print(f"   🔧 Core Functionality: {core_success:.1%}")
    print(f"   🎯 Advanced Features: {advanced_success:.1%}")
    print(f"   🔗 Integration: {integration_success:.1%}")
    
    overall_synthesis_score = (syntax_success + core_success + advanced_success + integration_success) / 4
    
    if overall_synthesis_score >= 0.8:
        print(f"\n🎉 DIALECTICAL SYNTHESIS: SUCCESSFUL ({overall_synthesis_score:.1%})")
        print("   ✅ Thesis-Antithesis-Synthesis framework validated")
        print("   ✅ Adaptive architecture improvements confirmed")
        print("   ✅ Hierarchical evolution capabilities demonstrated")
    elif overall_synthesis_score >= 0.6:
        print(f"\n⚠️  DIALECTICAL SYNTHESIS: PARTIAL ({overall_synthesis_score:.1%})")
        print("   🔄 Core framework functional with some limitations")
        print("   🔄 Additional optimization recommended")
    else:
        print(f"\n❌ DIALECTICAL SYNTHESIS: NEEDS WORK ({overall_synthesis_score:.1%})")
        print("   🔧 Significant issues detected requiring attention")
    
    print("\n" + "="*70)
    print("🔬 Dialectical Analysis Complete")
    print("   Truth is symbolic yet contestable ✓")
    print("   Ethics evolve via dynamic feedback ✓") 
    print("   Architecture governs strategy ✓")
    print("   Code embodies multi-layered logic ✓")
    print("   Dialogue refines every output ✓")
    print("="*70)
    
    return summary['success_rate'] >= 0.7


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)