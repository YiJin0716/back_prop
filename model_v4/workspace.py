"""Create an ordered W&B view without changing an existing user workspace."""
import json
from pathlib import Path
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr
from .monitor import COMPONENTS, GROUPS


def create_workspace(destination):
    panels = [wr.LinePlot(title='total',x='epoch',y=['00_training/total'])]
    panels += [wr.LinePlot(title=name,x='epoch',y=[f'01_components/{i:02d}_{name}'])
               for i,name in enumerate(COMPONENTS,1)]
    objectives = [wr.LinePlot(title=name,x='epoch',y=[f'02_objectives/{i:02d}_{name}'])
                  for i,name in enumerate(GROUPS,1)]
    workspace = ws.Workspace(entity='jin20020716-duke-university',project='imaging-feature',
        name='V4 and V4 oversample - epoch losses',auto_generate_panels=False,
        runset_settings=ws.RunsetSettings(query='v4-'),sections=[
            ws.Section(name='Official annotation warmup',is_open=True,panels=[
                wr.LinePlot(title='MedicalNet semantic warmup',x='warmup_epoch',y=['warmup/semantic']),
                wr.LinePlot(title='Official radiomics baseline fit',x='warmup_epoch',y=['warmup/baseline_loss']),
                wr.LinePlot(title='Official semantic residual fit',x='warmup_epoch',y=['warmup/residual_risk_loss'])]),
            ws.Section(name='Component losses',is_open=True,panels=panels),
            ws.Section(name='Joint objectives',is_open=True,panels=objectives)])
    workspace.save()
    data = dict(url=workspace.url,groups=list(GROUPS),components=list(COMPONENTS),
                x_axis='epoch',epoch_only=True)
    Path(destination).write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data))


if __name__=='__main__':
    import sys
    create_workspace(sys.argv[1])
