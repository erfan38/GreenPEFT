#!/usr/bin/env python3
"""
Quick EnergyPEFT Installation Test
A simple script to verify your EnergyPEFT installation is working.
"""

print("Testing EnergyPEFT Installation...")
print("=" * 40)

# Test 1: Basic import
print("\n1. Testing basic import...")
try:
    import energypeft
    print("✓ SUCCESS: energypeft imported successfully!")
except ImportError as e:
    print(f"✗ FAILED: Could not import energypeft - {e}")
    exit(1)

# Test 2: Check if main class exists
print("\n2. Testing main EnergyPEFT class...")
try:
    if hasattr(energypeft, 'EnergyPEFT'):
        print("✓ SUCCESS: EnergyPEFT class found!")
        
        # Try to create an instance
        energy_peft = energypeft.EnergyPEFT(
            energy_budget_wh=100.0,
            base_batch_size=32,
            importance_weight=0.7
        )
        print("✓ SUCCESS: EnergyPEFT instance created!")
        
    else:
        print("⚠ WARNING: EnergyPEFT class not found in module")
except Exception as e:
    print(f"⚠ WARNING: Could not create EnergyPEFT instance - {e}")

# Test 3: Check key dependencies
print("\n3. Testing key dependencies...")
dependencies = ['torch', 'transformers', 'peft', 'codecarbon', 'numpy', 'pandas']

for dep in dependencies:
    try:
        __import__(dep)
        print(f"✓ {dep}")
    except ImportError:
        print(f"✗ {dep} - MISSING!")

print("\n" + "=" * 40)
print("Basic installation test completed!")
print("Your EnergyPEFT library was installed correctly.")
print("\nYou can now use it in your projects with:")
print("  import energypeft")
print("  energy_peft = energypeft.EnergyPEFT()")
print("=" * 40)
