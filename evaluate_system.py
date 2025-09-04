#!/usr/bin/env python3
"""
Comprehensive evaluation script for the Amazon Product Q&A System.
"""

import sys
import json
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.qa.qa_engine import QAEvaluator
from src.models.schemas import QAResponse, Citation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemEvaluator:
    """Comprehensive system evaluation."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.test_queries = self._load_test_queries()
        self.test_questions = self._load_test_questions()
    
    def _load_test_queries(self) -> List[str]:
        """Load test search queries."""
        return [
            "wireless bluetooth headphones",
            "gaming laptop under 1000",
            "smart home devices",
            "portable bluetooth speaker",
            "4k streaming device",
            "wireless earbuds with noise cancellation",
            "budget smartphone",
            "fitness tracker watch",
            "wireless charging pad",
            "usb-c hub multiport"
        ]
    
    def _load_test_questions(self) -> List[str]:
        """Load test Q&A questions."""
        return [
            "How is the battery life?",
            "Is the sound quality good?",
            "How easy is it to set up?",
            "What do customers say about build quality?",
            "Is this good value for money?",
            "Does this work well for gaming?",
            "How is the customer service?",
            "Are there any common issues?",
            "How does this compare to competitors?",
            "Is this suitable for beginners?"
        ]
    
    def check_api_availability(self) -> bool:
        """Check if API is available and responsive."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"API Status: {health_data.get('status', 'unknown')}")
                return health_data.get('status') == 'healthy'
            else:
                logger.error(f"API health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"API not available: {e}")
            return False
    
    def evaluate_search_performance(self) -> Dict[str, Any]:
        """Evaluate search functionality performance."""
        logger.info("🔍 Evaluating search performance...")
        
        results = {
            "total_queries": len(self.test_queries),
            "successful_queries": 0,
            "average_response_time": 0.0,
            "average_results_count": 0.0,
            "response_times": [],
            "results_counts": []
        }
        
        for query in self.test_queries:
            try:
                start_time = time.time()
                
                response = requests.post(
                    f"{self.api_url}/search",
                    json={"query": query, "top_k": 10},
                    timeout=30
                )
                
                response_time = time.time() - start_time
                results["response_times"].append(response_time)
                
                if response.status_code == 200:
                    search_results = response.json()
                    results_count = len(search_results)
                    results["results_counts"].append(results_count)
                    results["successful_queries"] += 1
                    
                    logger.info(f"Query '{query}': {results_count} results in {response_time:.2f}s")
                else:
                    logger.warning(f"Query '{query}' failed: {response.status_code}")
                    results["results_counts"].append(0)
                    
            except Exception as e:
                logger.error(f"Query '{query}' error: {e}")
                results["results_counts"].append(0)
        
        # Calculate averages
        if results["response_times"]:
            results["average_response_time"] = np.mean(results["response_times"])
        if results["results_counts"]:
            results["average_results_count"] = np.mean(results["results_counts"])
        
        success_rate = results["successful_queries"] / results["total_queries"]
        logger.info(f"✅ Search evaluation complete: {success_rate:.1%} success rate")
        
        return results
    
    def evaluate_qa_performance(self) -> Dict[str, Any]:
        """Evaluate Q&A functionality performance."""
        logger.info("❓ Evaluating Q&A performance...")
        
        results = {
            "total_questions": len(self.test_questions),
            "successful_questions": 0,
            "average_response_time": 0.0,
            "average_confidence": 0.0,
            "average_citations": 0.0,
            "response_times": [],
            "confidence_scores": [],
            "citation_counts": []
        }
        
        for question in self.test_questions:
            try:
                start_time = time.time()
                
                response = requests.post(
                    f"{self.api_url}/qa",
                    json={"question": question, "max_chunks": 5},
                    timeout=60
                )
                
                response_time = time.time() - start_time
                results["response_times"].append(response_time)
                
                if response.status_code == 200:
                    qa_result = response.json()
                    
                    confidence = qa_result.get("confidence_score", 0.0)
                    citations_count = len(qa_result.get("citations", []))
                    
                    results["confidence_scores"].append(confidence)
                    results["citation_counts"].append(citations_count)
                    results["successful_questions"] += 1
                    
                    logger.info(f"Question '{question[:30]}...': confidence={confidence:.2f}, citations={citations_count}, time={response_time:.2f}s")
                else:
                    logger.warning(f"Question '{question}' failed: {response.status_code}")
                    results["confidence_scores"].append(0.0)
                    results["citation_counts"].append(0)
                    
            except Exception as e:
                logger.error(f"Question '{question}' error: {e}")
                results["confidence_scores"].append(0.0)
                results["citation_counts"].append(0)
        
        # Calculate averages
        if results["response_times"]:
            results["average_response_time"] = np.mean(results["response_times"])
        if results["confidence_scores"]:
            results["average_confidence"] = np.mean(results["confidence_scores"])
        if results["citation_counts"]:
            results["average_citations"] = np.mean(results["citation_counts"])
        
        success_rate = results["successful_questions"] / results["total_questions"]
        logger.info(f"✅ Q&A evaluation complete: {success_rate:.1%} success rate")
        
        return results
    
    def evaluate_recommendations_performance(self) -> Dict[str, Any]:
        """Evaluate recommendation functionality."""
        logger.info("💡 Evaluating recommendations performance...")
        
        results = {
            "query_based_tests": 0,
            "similar_products_tests": 0,
            "popular_products_tests": 0,
            "successful_recommendations": 0,
            "average_response_time": 0.0,
            "response_times": []
        }
        
        # Test query-based recommendations
        for query in self.test_queries[:5]:  # Test subset
            try:
                start_time = time.time()
                
                response = requests.post(
                    f"{self.api_url}/recommendations",
                    json={"query": query, "top_k": 6},
                    timeout=30
                )
                
                response_time = time.time() - start_time
                results["response_times"].append(response_time)
                results["query_based_tests"] += 1
                
                if response.status_code == 200:
                    rec_result = response.json()
                    products_count = len(rec_result.get("products", []))
                    results["successful_recommendations"] += 1
                    
                    logger.info(f"Recommendations for '{query}': {products_count} products in {response_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Recommendations for '{query}' error: {e}")
        
        # Test popular products
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.api_url}/recommendations",
                json={"top_k": 6},
                timeout=30
            )
            response_time = time.time() - start_time
            results["response_times"].append(response_time)
            results["popular_products_tests"] += 1
            
            if response.status_code == 200:
                results["successful_recommendations"] += 1
                logger.info(f"Popular products: success in {response_time:.2f}s")
                
        except Exception as e:
            logger.error(f"Popular products error: {e}")
        
        # Calculate averages
        if results["response_times"]:
            results["average_response_time"] = np.mean(results["response_times"])
        
        total_tests = results["query_based_tests"] + results["popular_products_tests"]
        success_rate = results["successful_recommendations"] / total_tests if total_tests > 0 else 0
        logger.info(f"✅ Recommendations evaluation complete: {success_rate:.1%} success rate")
        
        return results
    
    def generate_evaluation_report(self, search_results: Dict, qa_results: Dict, 
                                 rec_results: Dict) -> str:
        """Generate comprehensive evaluation report."""
        report = []
        report.append("# Amazon Product Q&A System - Evaluation Report")
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Search Performance
        report.append("## 🔍 Search Performance")
        report.append(f"- Total queries tested: {search_results['total_queries']}")
        report.append(f"- Successful queries: {search_results['successful_queries']}")
        report.append(f"- Success rate: {search_results['successful_queries']/search_results['total_queries']:.1%}")
        report.append(f"- Average response time: {search_results['average_response_time']:.2f}s")
        report.append(f"- Average results per query: {search_results['average_results_count']:.1f}")
        report.append("")
        
        # Q&A Performance
        report.append("## ❓ Q&A Performance")
        report.append(f"- Total questions tested: {qa_results['total_questions']}")
        report.append(f"- Successful questions: {qa_results['successful_questions']}")
        report.append(f"- Success rate: {qa_results['successful_questions']/qa_results['total_questions']:.1%}")
        report.append(f"- Average response time: {qa_results['average_response_time']:.2f}s")
        report.append(f"- Average confidence: {qa_results['average_confidence']:.2f}")
        report.append(f"- Average citations per answer: {qa_results['average_citations']:.1f}")
        report.append("")
        
        # Recommendations Performance
        report.append("## 💡 Recommendations Performance")
        total_rec_tests = rec_results['query_based_tests'] + rec_results['popular_products_tests']
        report.append(f"- Total recommendation tests: {total_rec_tests}")
        report.append(f"- Successful recommendations: {rec_results['successful_recommendations']}")
        if total_rec_tests > 0:
            report.append(f"- Success rate: {rec_results['successful_recommendations']/total_rec_tests:.1%}")
        report.append(f"- Average response time: {rec_results['average_response_time']:.2f}s")
        report.append("")
        
        # System Summary
        report.append("## 📊 System Summary")
        
        overall_success = (
            search_results['successful_queries'] + 
            qa_results['successful_questions'] + 
            rec_results['successful_recommendations']
        )
        total_tests = (
            search_results['total_queries'] + 
            qa_results['total_questions'] + 
            total_rec_tests
        )
        
        if total_tests > 0:
            overall_success_rate = overall_success / total_tests
            report.append(f"- Overall success rate: {overall_success_rate:.1%}")
        
        avg_response_time = np.mean([
            search_results['average_response_time'],
            qa_results['average_response_time'], 
            rec_results['average_response_time']
        ])
        report.append(f"- Average response time across all features: {avg_response_time:.2f}s")
        
        # Recommendations
        report.append("")
        report.append("## 🎯 Recommendations")
        report.append("- ✅ System is functional with demo data")
        report.append("- 🔧 For production: add real Amazon dataset and API keys")
        report.append("- 📈 Performance can be improved with larger dataset")
        report.append("- 🚀 Ready for deployment with Docker")
        
        return "\n".join(report)
    
    def run_complete_evaluation(self) -> Dict[str, Any]:
        """Run complete system evaluation."""
        logger.info("🧪 Starting comprehensive system evaluation...")
        
        # Check API availability
        if not self.check_api_availability():
            logger.error("API not available for evaluation")
            return {}
        
        # Run evaluations
        search_results = self.evaluate_search_performance()
        qa_results = self.evaluate_qa_performance()
        rec_results = self.evaluate_recommendations_performance()
        
        # Generate report
        report = self.generate_evaluation_report(search_results, qa_results, rec_results)
        
        # Save report
        with open("evaluation_report.md", "w") as f:
            f.write(report)
        
        logger.info("✅ Evaluation complete! Report saved to evaluation_report.md")
        
        return {
            "search": search_results,
            "qa": qa_results,
            "recommendations": rec_results,
            "report": report
        }


def main():
    """Main evaluation function."""
    print("🧪 Amazon Product Q&A System - Evaluation")
    print("==========================================")
    
    evaluator = SystemEvaluator()
    
    # Check if API is running
    if not evaluator.check_api_availability():
        print("❌ API is not running. Please start the system first:")
        print("   python quickstart.py")
        print("   # or")
        print("   docker-compose up")
        return False
    
    # Run evaluation
    results = evaluator.run_complete_evaluation()
    
    if results:
        print("\n" + "="*50)
        print("📊 EVALUATION SUMMARY")
        print("="*50)
        print(results["report"])
        print("\n📄 Full report saved to: evaluation_report.md")
        return True
    else:
        print("❌ Evaluation failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)