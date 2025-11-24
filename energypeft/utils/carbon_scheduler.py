# carbon_scheduler.py
from codecarbon import EmissionsTracker
import time
import requests

class CarbonScheduler:
    def __init__(self, region=None, carbon_threshold=300, check_interval=900, project_name="carbon_aware_training", electricity_maps_token=None):
        """
        region: ISO country code (e.g., 'US-NY'). If None, auto-detects.
        carbon_threshold: Only run if grid < this (gCO2/kWh)
        check_interval: How often to re-check (seconds)
        project_name: For CodeCarbon logs
        electricity_maps_token: API token for Electricity Maps (optional)
        """
        self.region = region
        self.carbon_threshold = carbon_threshold
        self.check_interval = check_interval
        self.project_name = project_name
        self.electricity_maps_token = electricity_maps_token
        self.tracker = None

    def get_current_carbon_intensity(self):
        """Get current grid carbon intensity using Electricity Maps API."""
        if not self.electricity_maps_token or not self.region:
            print("[CarbonScheduler] No API token or region provided. Skipping carbon intensity check.")
            return None
        url = f"https://api.electricitymap.org/v3/carbon-intensity/latest?zone={self.region}"
        headers = {"auth-token": self.electricity_maps_token}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                ci = data.get("carbonIntensity", None)
                return ci
            else:
                print(f"[CarbonScheduler] API error: {response.status_code}")
                return None
        except Exception as e:
            print(f"[CarbonScheduler] Exception fetching carbon intensity: {e}")
            return None

    def wait_for_green_grid(self):
        """Wait until grid carbon intensity is below threshold (if API available)."""
        if not self.electricity_maps_token or not self.region:
            print("[CarbonScheduler] No carbon intensity API configured. Proceeding without scheduling.")
            return
        print(f"[CarbonScheduler] Waiting for grid to be green (<{self.carbon_threshold} gCO2/kWh)...")
        while True:
            ci = self.get_current_carbon_intensity()
            if ci is not None:
                print(f"[CarbonScheduler] Current grid carbon intensity: {ci:.1f} gCO2/kWh")
                if ci < self.carbon_threshold:
                    print("[CarbonScheduler] Grid is green enough. Starting training!")
                    break
            else:
                print("[CarbonScheduler] Could not fetch carbon intensity. Proceeding anyway.")
                break
            time.sleep(self.check_interval)

    def __enter__(self):
        self.wait_for_green_grid()
        self.tracker = EmissionsTracker(
            project_name=self.project_name,
            country_iso_code=self.region,
            save_to_file=True
        )
        self.tracker.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        emissions = self.tracker.stop()
        print(f"[CarbonScheduler] Training complete. Estimated CO2 emissions: {emissions:.4f} kgCO2e")

    def run(self, training_fn, *args, **kwargs):
        """Run a training function with carbon-aware scheduling and emissions tracking."""
        with self:
            return training_fn(*args, **kwargs)