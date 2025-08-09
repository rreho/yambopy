#!/usr/bin/env python3
"""
Script to build the QuREX-book documentation.

This script:
1. Generates auto-generated API documentation
2. Builds the Jupyter Book
3. Provides helpful output about the build process

Usage:
    python build_docs.py [--clean]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        if result.stdout:
            print("Output:")
            print(result.stdout)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e}")
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description='Build QuREX-book documentation')
    parser.add_argument('--clean', action='store_true', 
                       help='Clean build directory before building')
    parser.add_argument('--api-only', action='store_true',
                       help='Only generate API documentation')
    args = parser.parse_args()
    
    # Change to docs directory
    docs_dir = Path(__file__).parent
    os.chdir(docs_dir)
    
    print("🚀 QuREX-book Documentation Build Process")
    print(f"📁 Working directory: {docs_dir}")
    
    # Step 1: Generate API documentation
    if not run_command("python generate_api_docs.py", 
                      "Generating ExcitonGroupTheory API documentation"):
        print("⚠️  ExcitonGroupTheory API documentation generation failed, but continuing...")
    
    # Step 1b: Generate comprehensive API documentation
    if not run_command("python generate_comprehensive_api_docs.py", 
                      "Generating comprehensive API documentation"):
        print("⚠️  Comprehensive API documentation generation failed, but continuing...")
    
    if args.api_only:
        print("\n✅ API documentation generation completed!")
        return
    
    # Step 2: Clean if requested
    if args.clean:
        if not run_command("jupyter-book clean .", "Cleaning build directory"):
            print("⚠️  Clean failed, but continuing...")
    
    # Step 3: Build the book
    if not run_command("jupyter-book build .", "Building Jupyter Book"):
        print("❌ Documentation build failed!")
        sys.exit(1)
    
    # Step 4: Success message
    print(f"\n{'='*60}")
    print("🎉 DOCUMENTATION BUILD COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"📖 Documentation available at: {docs_dir}/_build/html/index.html")
    print(f"🌐 Open in browser: file://{docs_dir.absolute()}/_build/html/index.html")
    print("\n📝 What was built:")
    print("  ✅ Auto-generated API documentation from docstrings")
    print("  ✅ Theoretical background pages")
    print("  ✅ Tutorial and example notebooks")
    print("  ✅ Complete QuREX-book with navigation")
    
    print(f"\n🔄 To update documentation:")
    print(f"  • Modify docstrings in source code")
    print(f"  • Run: python build_docs.py")
    print(f"  • API docs will be automatically regenerated")

if __name__ == '__main__':
    main()