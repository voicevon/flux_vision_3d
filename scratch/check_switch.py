import cv2
from tools.asparagus_offline import AsparagusOfflineApp

app = AsparagusOfflineApp()
print("sel_idx:", app.sel_idx, "sample:", app.samples[app.sel_idx]["name"])
app.run_analyze()
print("Pipe A:", len(app.targets))
app.switch_pipeline("edge_centerline")
print("Pipe B targets:", len(app.targets), "result targets:", len(app.pipeline_result.targets) if app.pipeline_result else None)
