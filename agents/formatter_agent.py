import os
from typing import Dict, Any
from utils.config_loader import ConfigLoader
from utils.local_client import LocalClient

class FormatterAgent:
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.formatter_config = config.get_agent_config('formatter')
        self.local_config = config.config['inference']['local']
        
        # Check if fine-tuned model exists, otherwise use base model
        fine_tuned_path = self.formatter_config.get('fine_tuned_path')
        if fine_tuned_path and os.path.exists(fine_tuned_path):
            model_name = fine_tuned_path
            print("-------------------- Using fine-tuned model --------------------")
        else:
            model_name = self.local_config['formatter_model']
            print("Using base model")
        
        # Initialize local model client
        self.local_client = LocalClient(
            model_name=model_name,
            device=self.local_config['device'],
            temperature=self.local_config['temperature'],
            max_tokens=self.local_config['max_tokens']
        )
        
        # System prompt for LaTeX formatting
        self.latex_system_prompt = """You are an expert at converting academic solutions to properly formatted LaTeX. Follow these rules:

1. Format ALL mathematical expressions using $...$ for inline math and $$...$$ for display math
2. Use appropriate LaTeX environments: equation, align, itemize, enumerate
3. Structure with sections using \\section, \\subsection
4. Use \\textbf for bold, \\textit for italics
5. Ensure the output compiles without errors
6. Return ONLY the LaTeX code without explanations

Example:
Input: "Solve x^2 + 1 = 0. Solution: x = ±i"
Output: "Solve $x^2 + 1 = 0$. Solution: $x = \\pm i$"

Now convert the following:"""

    def text_to_latex(self, text_content: str, document_type: str = "homework") -> Dict[str, Any]:
        """Convert text content to LaTeX using local model"""
        prompt = f"Convert this {document_type} solution to LaTeX:\n\n{text_content}"
        
        try:
            print("🔄 Generating LaTeX...")
            latex_code = self.local_client.generate_response(prompt, self.latex_system_prompt)
            
            # Validation
            is_valid = self._validate_latex(latex_code)
            warnings = self._get_latex_warnings(latex_code)
            
            return {
                "latex_code": latex_code,
                "is_valid": is_valid,
                "model_used": "fine_tuned" if "fine_tuned" in str(self.local_client.model_name) else "base",
                "warnings": warnings
            }
            
        except Exception as e:
            return {
                "latex_code": "",
                "is_valid": False,
                "error": str(e),
                "model_used": "fine_tuned" if "fine_tuned" in str(self.local_client.model_name) else "base"
            }
    def get_supported_types(self):
        """Return supported document types"""
        return ["homework", "algorithm", "proof", "general", "explanation"]
    
    def is_ready(self):
        """Check if formatter agent is ready"""
        return self.local_client is not None and self.local_client.model is not None
    def _validate_latex(self, latex_code: str) -> bool:
        """Validate LaTeX code"""
        if not latex_code or latex_code.strip() == "":
            return False
        return any(cmd in latex_code for cmd in ['$', '\\begin', '\\section', '\\documentclass'])
    
    def _get_latex_warnings(self, latex_code: str) -> list:
        """Get LaTeX warnings"""
        warnings = []
        if latex_code.count('{') != latex_code.count('}'):
            warnings.append("Unbalanced braces")
        if '$$' in latex_code and latex_code.count('$$') % 2 != 0:
            warnings.append("Unbalanced display math")
        return warnings