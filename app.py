"""
Entrypoint wrapper for the SmartLoan Intelligence System.
Allows running via either:
    streamlit run app.py
or:
    streamlit run loan_approval_system.py
"""
import runpy

if __name__ == "__main__":
    runpy.run_module("loan_approval_system", run_name="__main__")
