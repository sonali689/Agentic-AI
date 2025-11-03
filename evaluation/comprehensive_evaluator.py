#!/usr/bin/env python3
"""
Comprehensive evaluation for both RAG and Formatter agents
"""

import json
import os
from typing import Dict, List, Any
from utils.config_loader import ConfigLoader
from agents.rag_agent import RAGAgent
from agents.formatter_agent import FormatterAgent
import pandas as pd
from datetime import datetime
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(ROOT)
class ComprehensiveEvaluator:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.rag_agent = RAGAgent(config)
        self.formatter_agent = FormatterAgent(config)
        
    def evaluate_rag_agent(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """Comprehensive RAG evaluation"""
        print("🧪 Evaluating RAG Agent...")
        
        # Build knowledge base first
        self.rag_agent.build_knowledge_base("./test_courses/CS101", "CS101")
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"Testing RAG case {i+1}/{len(test_cases)}: {test_case['question'][:50]}...")
            
            try:
                # Get answer from RAG
                rag_result = self.rag_agent.query(test_case['question'])
                
                # Calculate metrics
                metrics = self._calculate_rag_metrics(
                    rag_result['answer'], 
                    test_case.get('expected_answer', ''),
                    test_case.get('expected_keywords', []),
                    rag_result['sources']
                )
                
                result = {
                    'test_case_id': i,
                    'question': test_case['question'],
                    'answer': rag_result['answer'],
                    'sources': rag_result['sources'],
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
                
            except Exception as e:
                print(f"❌ Error in test case {i}: {e}")
                results.append({
                    'test_case_id': i,
                    'question': test_case['question'],
                    'error': str(e),
                    'metrics': {'overall_score': 0}
                })
        
        # Calculate aggregate metrics
        aggregate_metrics = self._aggregate_rag_metrics(results)
        
        return {
            'evaluation_type': 'rag_agent',
            'timestamp': datetime.now().isoformat(),
            'total_test_cases': len(test_cases),
            'successful_cases': len([r for r in results if 'error' not in r]),
            'aggregate_metrics': aggregate_metrics,
            'detailed_results': results
        }
    
    def evaluate_formatter_agent(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """Comprehensive Formatter evaluation"""
        print("🧪 Evaluating Formatter Agent...")
        
        results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"Testing Formatter case {i+1}/{len(test_cases)}")
            
            try:
                # Generate LaTeX
                formatter_result = self.formatter_agent.text_to_latex(
                    test_case['input_text'],
                    test_case.get('document_type', 'homework')
                )
                
                # Calculate metrics
                metrics = self._calculate_formatter_metrics(
                    formatter_result['latex_code'],
                    test_case.get('expected_latex_elements', []),
                    test_case['input_text']
                )
                
                result = {
                    'test_case_id': i,
                    'input_text': test_case['input_text'],
                    'latex_output': formatter_result['latex_code'],
                    'is_valid': formatter_result['is_valid'],
                    'model_used': formatter_result['model_used'],
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
                
            except Exception as e:
                print(f"❌ Error in test case {i}: {e}")
                results.append({
                    'test_case_id': i,
                    'input_text': test_case['input_text'],
                    'error': str(e),
                    'metrics': {'overall_score': 0}
                })
        
        # Calculate aggregate metrics
        aggregate_metrics = self._aggregate_formatter_metrics(results)
        
        return {
            'evaluation_type': 'formatter_agent',
            'timestamp': datetime.now().isoformat(),
            'total_test_cases': len(test_cases),
            'successful_cases': len([r for r in results if 'error' not in r]),
            'aggregate_metrics': aggregate_metrics,
            'detailed_results': results
        }
    
    def _calculate_rag_metrics(self, answer: str, expected_answer: str, 
                             expected_keywords: List[str], sources: List[str]) -> Dict[str, float]:
        """Calculate RAG quality metrics"""
        metrics = {}
        
        # Answer relevance (keyword-based)
        if expected_keywords:
            keyword_hits = sum(1 for keyword in expected_keywords if keyword.lower() in answer.lower())
            metrics['keyword_precision'] = keyword_hits / len(expected_keywords) if expected_keywords else 0.0
        
        # Answer length appropriateness
        answer_length = len(answer.split())
        if answer_length < 10:
            metrics['length_score'] = 0.3
        elif answer_length < 50:
            metrics['length_score'] = 0.7
        else:
            metrics['length_score'] = 1.0
        
        # Source relevance
        metrics['source_relevance'] = 1.0 if len(sources) > 0 else 0.0
        
        # Overall score (weighted average)
        weights = {'keyword_precision': 0.4, 'length_score': 0.3, 'source_relevance': 0.3}
        metrics['overall_score'] = sum(metrics.get(k, 0) * weights[k] for k in weights)
        
        return metrics
    
    def _calculate_formatter_metrics(self, latex_code: str, expected_elements: List[str], 
                                   original_text: str) -> Dict[str, float]:
        """Calculate Formatter quality metrics"""
        metrics = {}
        
        # LaTeX validity
        metrics['compilation_likelihood'] = self._estimate_compilation_score(latex_code)
        
        # Structure quality
        metrics['structure_score'] = self._calculate_structure_score(latex_code)
        
        # Math formatting
        metrics['math_formatting_score'] = self._calculate_math_score(latex_code, original_text)
        
        # Expected elements
        if expected_elements:
            element_hits = sum(1 for elem in expected_elements if elem in latex_code)
            metrics['element_completeness'] = element_hits / len(expected_elements)
        else:
            metrics['element_completeness'] = 0.5  # Neutral score
        
        # Overall score
        weights = {
            'compilation_likelihood': 0.3,
            'structure_score': 0.25,
            'math_formatting_score': 0.25,
            'element_completeness': 0.2
        }
        metrics['overall_score'] = sum(metrics[k] * weights[k] for k in weights)
        
        return metrics
    
    def _estimate_compilation_score(self, latex_code: str) -> float:
        """Estimate LaTeX compilation likelihood"""
        score = 0.0
        
        # Basic structure checks
        if '\\documentclass' in latex_code:
            score += 0.3
        if '\\begin{document}' in latex_code:
            score += 0.3
        if '\\end{document}' in latex_code:
            score += 0.2
        
        # Balance checks
        if latex_code.count('{') == latex_code.count('}'):
            score += 0.2
        else:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_structure_score(self, latex_code: str) -> float:
        """Calculate LaTeX structure quality"""
        score = 0.0
        
        # Sections
        if any(cmd in latex_code for cmd in ['\\section', '\\subsection']):
            score += 0.3
        
        # Environments
        environments = ['equation', 'align', 'itemize', 'enumerate', 'table']
        env_count = sum(1 for env in environments if f'\\begin{{{env}}}' in latex_code)
        score += min(0.3, env_count * 0.1)
        
        # Formatting commands
        formatting_cmds = ['\\textbf', '\\textit', '\\underline']
        fmt_count = sum(1 for cmd in formatting_cmds if cmd in latex_code)
        score += min(0.2, fmt_count * 0.05)
        
        # Document completeness
        if len(latex_code.strip()) > 100:
            score += 0.2
            
        return score
    
    def _calculate_math_score(self, latex_code: str, original_text: str) -> float:
        """Calculate math formatting quality"""
        # Count math environments in LaTeX
        inline_math = latex_code.count('$') // 2  # Pairs of $
        display_math = latex_code.count('$$') // 2
        equation_env = latex_code.count('\\begin{equation')
        
        total_math_elements = inline_math + display_math + equation_env
        
        # Estimate math content in original text
        math_indicators = ['=', '+', '-', '*', '/', '^', 'sqrt', 'frac', 'sum', 'int']
        original_math_count = sum(1 for indicator in math_indicators if indicator in original_text)
        
        if original_math_count > 0:
            conversion_ratio = total_math_elements / original_math_count
            return min(1.0, conversion_ratio)
        else:
            return 0.5  # Neutral score if no math expected
    
    def _aggregate_rag_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate RAG metrics"""
        successful_results = [r for r in results if 'metrics' in r]
        
        if not successful_results:
            return {'overall_score': 0.0}
        
        aggregates = {}
        for metric in successful_results[0]['metrics'].keys():
            values = [r['metrics'][metric] for r in successful_results]
            aggregates[f'avg_{metric}'] = sum(values) / len(values)
            aggregates[f'min_{metric}'] = min(values)
            aggregates[f'max_{metric}'] = max(values)
        
        return aggregates
    
    def _aggregate_formatter_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate Formatter metrics"""
        successful_results = [r for r in results if 'metrics' in r and r['is_valid']]
        
        if not successful_results:
            return {'overall_score': 0.0}
        
        aggregates = {}
        for metric in successful_results[0]['metrics'].keys():
            values = [r['metrics'][metric] for r in successful_results]
            aggregates[f'avg_{metric}'] = sum(values) / len(values)
            aggregates[f'min_{metric}'] = min(values)
            aggregates[f'max_{metric}'] = max(values)
        
        return aggregates
    
    def save_evaluation_results(self, results: Dict[str, Any], filename: str):
        """Save evaluation results to JSON"""
        os.makedirs('./evaluation_results', exist_ok=True)
        filepath = f'./evaluation_results/{filename}'
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Evaluation results saved to: {filepath}")
        
        # Also save as CSV for easier analysis
        self._save_detailed_csv(results, filename.replace('.json', '_detailed.csv'))
    
    def _save_detailed_csv(self, results: Dict[str, Any], filename: str):
        """Save detailed results as CSV"""
        if 'detailed_results' not in results:
            return
        
        df_data = []
        for result in results['detailed_results']:
            row = {'test_case_id': result['test_case_id']}
            
            if 'question' in result:
                row['type'] = 'rag'
                row['input'] = result['question'][:100]  # Truncate for CSV
                row['output'] = result['answer'][:100] if 'answer' in result else 'ERROR'
            else:
                row['type'] = 'formatter'
                row['input'] = result['input_text'][:100] if 'input_text' in result else 'ERROR'
                row['output'] = result['latex_output'][:100] if 'latex_output' in result else 'ERROR'
            
            if 'metrics' in result:
                for metric, value in result['metrics'].items():
                    row[metric] = value
            
            df_data.append(row)
        
        if df_data:
            df = pd.DataFrame(df_data)
            df.to_csv(f'./evaluation_results/{filename}', index=False)
            print(f"✅ Detailed CSV saved to: ./evaluation_results/{filename}")

def create_test_cases():
    """Create comprehensive test cases for evaluation"""
    
    # RAG Test Cases
    rag_test_cases = [
        {
            'question': 'What is the difference between DFS and BFS?',
            'expected_keywords': ['DFS', 'BFS', 'stack', 'queue', 'depth', 'breadth'],
            'expected_answer': 'DFS uses stack, BFS uses queue'
        },
        {
            'question': 'What is the time complexity of graph traversal algorithms?',
            'expected_keywords': ['O(V+E)', 'vertices', 'edges', 'complexity'],
            'expected_answer': 'O(V + E) where V is vertices and E is edges'
        },
        {
            'question': 'Explain how depth-first search works',
            'expected_keywords': ['stack', 'backtracking', 'recursive', 'LIFO'],
            'expected_answer': 'DFS explores as far as possible along each branch'
        }
    ]
    
    # Formatter Test Cases
    formatter_test_cases = [
        {
            'input_text': 'Solve the equation: x^2 + 2x + 1 = 0. Solution: (x+1)^2 = 0, so x = -1.',
            'document_type': 'homework',
            'expected_latex_elements': ['$', 'x^2', '=', '^2']
        },
        {
            'input_text': 'Algorithm: Binary Search. Steps: 1. Start with sorted array 2. Find middle element 3. Compare with target 4. Repeat on appropriate half.',
            'document_type': 'algorithm', 
            'expected_latex_elements': ['\\begin{algorithm}', '\\item', '\\end{algorithm}']
        },
        {
            'input_text': 'The derivative of f(x) = 3x^2 is f\'(x) = 6x. The integral is ∫3x^2 dx = x^3 + C.',
            'document_type': 'calculus',
            'expected_latex_elements': ['$', 'f(x)', '\\frac', '\\int']
        }
    ]
    
    return rag_test_cases, formatter_test_cases

def main():
    """Run comprehensive evaluation"""
    print("🚀 Starting Comprehensive Evaluation")
    print("=" * 60)
    
    config = ConfigLoader()
    evaluator = ComprehensiveEvaluator(config)
    
    # Create test cases
    rag_test_cases, formatter_test_cases = create_test_cases()
    
    # Evaluate RAG Agent
    print("\n1. Evaluating RAG Agent...")
    rag_results = evaluator.evaluate_rag_agent(rag_test_cases)
    evaluator.save_evaluation_results(rag_results, 'rag_evaluation.json')
    
    # Evaluate Formatter Agent  
    print("\n2. Evaluating Formatter Agent...")
    formatter_results = evaluator.evaluate_formatter_agent(formatter_test_cases)
    evaluator.save_evaluation_results(formatter_results, 'formatter_evaluation.json')
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    
    rag_metrics = rag_results['aggregate_metrics']
    formatter_metrics = formatter_results['aggregate_metrics']
    
    print(f"RAG Agent - Overall Score: {rag_metrics.get('avg_overall_score', 0):.2f}")
    print(f"Formatter Agent - Overall Score: {formatter_metrics.get('avg_overall_score', 0):.2f}")
    
    print(f"\nRAG Success Rate: {rag_results['successful_cases']}/{rag_results['total_test_cases']}")
    print(f"Formatter Success Rate: {formatter_results['successful_cases']}/{formatter_results['total_test_cases']}")
    
    # Generate final report
    generate_final_report(rag_results, formatter_results)

def generate_final_report(rag_results, formatter_results):
    """Generate final evaluation report"""
    report = {
        'evaluation_report': {
            'timestamp': datetime.now().isoformat(),
            'evaluation_framework': 'Comprehensive Agent Evaluation',
            'rag_agent': {
                'total_test_cases': rag_results['total_test_cases'],
                'successful_cases': rag_results['successful_cases'],
                'success_rate': rag_results['successful_cases'] / rag_results['total_test_cases'],
                'aggregate_metrics': rag_results['aggregate_metrics']
            },
            'formatter_agent': {
                'total_test_cases': formatter_results['total_test_cases'],
                'successful_cases': formatter_results['successful_cases'], 
                'success_rate': formatter_results['successful_cases'] / formatter_results['total_test_cases'],
                'aggregate_metrics': formatter_results['aggregate_metrics']
            }
        }
    }
    
    os.makedirs('./evaluation_results', exist_ok=True)
    with open('./evaluation_results/final_evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✅ Final evaluation report saved to: ./evaluation_results/final_evaluation_report.json")

if __name__ == "__main__":
    main()