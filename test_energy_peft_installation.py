#!/usr/bin/env python3
"""
EnergyPEFT Installation Verification Test
This script tests the basic functionality of your installed EnergyPEFT library.
"""

def test_basic_import():
    """Test basic import functionality"""
    print("=== Testing Basic Import ===")
    try:
        import energypeft
        print("✓ Successfully imported energypeft")
        
        # Check if version is available
        if hasattr(energypeft, '__version__'):
            print(f"✓ Version: {energypeft.__version__}")
        else:
            print("⚠ Version not found, but import successful")
        
        return True
    except ImportError as e:
        print(f"✗ Failed to import energypeft: {e}")
        return False

def test_core_components():
    """Test core component imports"""
    print("\n=== Testing Core Components ===")
    
    modules_to_test = [
        ('energy_monitor', 'energypeft.core.energy_monitor'),
        ('smart_sampler', 'energypeft.core.smart_sampler'),
        ('adaptive_batcher', 'energypeft.core.adaptive_batcher'),
        ('early_stopper', 'energypeft.core.early_stopper')
    ]
    
    success_count = 0
    for name, module_path in modules_to_test:
        try:
            module = __import__(module_path, fromlist=[name])
            print(f"✓ Successfully imported {name}")
            success_count += 1
        except ImportError as e:
            print(f"⚠ {name} import issue: {e}")
    
    return success_count

def test_integrations():
    """Test integration imports"""
    print("\n=== Testing Integration Components ===")
    
    integrations = [
        ('llamafactory', 'energypeft.integrations.llamafactory'),
        ('huggingface_peft', 'energypeft.integrations.huggingface_peft'),
        ('transformers', 'energypeft.integrations.transformers')
    ]
    
    success_count = 0
    for name, module_path in integrations:
        try:
            module = __import__(module_path, fromlist=[name])
            print(f"✓ Successfully imported {name} integration")
            success_count += 1
        except ImportError as e:
            print(f"⚠ {name} integration import issue: {e}")
    
    return success_count

def test_main_class():
    """Test main EnergyPEFT class instantiation"""
    print("\n=== Testing Main EnergyPEFT Class ===")
    try:
        import energypeft
        
        # Try to instantiate the main class
        energy_peft = energypeft.EnergyPEFT(
            energy_budget_wh=50.0,
            base_batch_size=16,
            importance_weight=0.8
        )
        print(" Successfully created EnergyPEFT instance")
        print(f"  - Energy budget: 50.0 Wh")
        print(f"  - Base batch size: 16")
        print(f"  - Importance weight: 0.8")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create EnergyPEFT instance: {e}")
        return False

def test_dependencies():
    """Test required dependencies"""
    print("\n=== Testing Dependencies ===")
    
    dependencies = [
        'torch',
        'transformers', 
        'peft',
        'datasets',
        'pynvml',
        'psutil',
        'codecarbon',
        'numpy',
        'pandas',
        'tqdm'
    ]
    
    success_count = 0
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep}")
            success_count += 1
        except ImportError:
            print(f"✗ {dep} - Missing!")
    
    print(f"\nDependencies: {success_count}/{len(dependencies)} available")
    return success_count == len(dependencies)

def main():
    """Run all tests"""
    print("EnergyPEFT Installation Verification")
    print("=" * 50)
    
    tests = [
        ("Basic Import", test_basic_import),
        ("Dependencies", test_dependencies),
        ("Core Components", test_core_components),
        ("Integrations", test_integrations),
        ("Main Class", test_main_class)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        results.append(test_func())
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print(" All tests passed! EnergyPEFT is ready to use!")
    elif passed >= total // 2:
        print("⚠ Partial success. Some components may need attention.")
    else:
        print(" Multiple issues detected. Check installation and file structure.")
    
    return passed == total

if __name__ == "__main__":
    main()