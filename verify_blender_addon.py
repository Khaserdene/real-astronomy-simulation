import sys
import os
import subprocess
import shutil

def main():
    # 1. Create dummy simulation
    print("Generating test simulation...")
    import core.engine
    from core.scenarios import build_scenario
    
    out_dir = os.path.join("output", "blender_test")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    
    core.engine.init_taichi("cpu")
    engine, dt = build_scenario('disk', n=200, seed=7, overrides=None, gravity_mode="direct", theta=0.6, engine_params={})
    
    engine.to_state().save(os.path.join(out_dir, "snap_0000.h5"))
    engine.step(dt)
    engine.to_state().save(os.path.join(out_dir, "snap_0001.h5"))
    engine.step(dt)
    engine.to_state().save(os.path.join(out_dir, "snap_0002.h5"))
    
    # 2. Check for Blender
    from gui.blender_launcher import find_blender
    blender_exe = find_blender()
    if not blender_exe:
        print("BLENDER_NOT_FOUND: Could not find Blender to run the test.")
        print("Please run this test manually after installing Blender 4.2+.")
        sys.exit(0)  # Graceful skip
        
    print(f"Found Blender: {blender_exe}")
    
    # 3. Create test script
    test_script = """
import sys
import os
sys.path.insert(0, r'd:/real_astronomy_sim')
sys.path.insert(0, r'd:/real_astronomy_sim/blender_addon')

try:
    import h5py
except ImportError:
    import site
    sys.path.append(site.getusersitepackages())
    try:
        import h5py
    except ImportError:
        print('BLENDER_TEST_FAIL: h5py not found in Blender Python')
        sys.exit(0)

import bpy
from blender_addon.loader import load_folder

sim_folder = sys.argv[sys.argv.index('--') + 1]

try:
    n_frames = load_folder(sim_folder)
    
    names = {o.name for o in bpy.data.objects}
    assert 'AstroStars' in names, f'Missing AstroStars, got {names}'
    
    assert bpy.context.scene.frame_end >= 2, f'Frame end too low: {bpy.context.scene.frame_end}'
    
    bpy.context.scene.frame_set(1)
    stars1 = len(bpy.data.objects['AstroStars'].data.vertices)
    bpy.context.scene.frame_set(2)
    stars2 = len(bpy.data.objects['AstroStars'].data.vertices)
    
    assert stars1 > 0, 'Frame 1 has no vertices'
    assert stars2 > 0, 'Frame 2 has no vertices'
    
    print('BLENDER_TEST_PASS')
except Exception as e:
    print(f'BLENDER_TEST_FAIL: {e}')
    import traceback
    traceback.print_exc()
"""
    script_path = os.path.join(out_dir, "test_script.py")
    with open(script_path, "w") as f:
        f.write(test_script)
        
    print("Running Blender in headless mode...")
    result = subprocess.run([
        blender_exe, "--background", "--python", script_path, "--", out_dir
    ], capture_output=True, text=True)
    
    out = result.stdout + result.stderr
    if "BLENDER_TEST_PASS" in out:
        print("PASS: Blender addon successfully loaded the simulation and created objects.")
    else:
        print("FAIL: Blender test failed.")
        print("Blender Output:")
        print(out)
        sys.exit(1)

if __name__ == '__main__':
    main()
