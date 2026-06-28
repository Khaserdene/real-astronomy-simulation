import os
import sys

def main():
    print("Testing realistic galaxy ICs...")
    
    from core.engine import init_taichi
    init_taichi()
    
    from core.scenarios import build_scenario
    
    # 1. Realistic Galaxy (Live Halo)
    print("\n--- Realistic Galaxy (Live Halo) ---")
    eng_live, dt1 = build_scenario("realistic_galaxy", n=5000, seed=42, overrides={"live_halo": True})
    print(f"Generated {eng_live.n} particles.")
    print("Stepping...")
    eng_live.step(dt1)
    
    # Check species distribution
    species = eng_live._f["is_star"].to_numpy()
    print(f"SMBH/Stars: {(species == 1).sum()}, Gas: {(species == 0).sum()}, DM: {(species == 2).sum()}")
    
    # 2. Realistic Galaxy (Analytic Halo)
    print("\n--- Realistic Galaxy (Analytic Halo) ---")
    eng_analytic, dt2 = build_scenario("realistic_galaxy", n=5000, seed=43, overrides={"live_halo": False})
    print(f"Generated {eng_analytic.n} particles.")
    print("Stepping...")
    eng_analytic.step(dt2)
    
    species = eng_analytic._f["is_star"].to_numpy()
    print(f"SMBH/Stars: {(species == 1).sum()}, Gas: {(species == 0).sum()}, DM: {(species == 2).sum()}")
    assert (species == 2).sum() == 0, "Analytic halo should not have DM particles"
    
    # 3. Proto Galaxy
    print("\n--- Proto Galaxy Collapse ---")
    eng_proto, dt3 = build_scenario("proto_galaxy_collapse", n=5000, seed=44, overrides={"live_halo": True})
    print(f"Generated {eng_proto.n} particles.")
    print("Stepping...")
    eng_proto.step(dt3)
    
    species = eng_proto._f["is_star"].to_numpy()
    # Proto galaxy should only have SMBH (star) and Gas and DM, but no Bulge/Disk stars initially
    print(f"SMBH/Stars: {(species == 1).sum()}, Gas: {(species == 0).sum()}, DM: {(species == 2).sum()}")
    assert (species == 1).sum() >= 1, "Proto galaxy should at least have the SMBH"
    
    print("\nALL PASSED.")

if __name__ == "__main__":
    main()
