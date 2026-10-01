"""Report-only numeric assets, derived from immutable experiment CSV/JSON files.

Run from the project root; --check verifies without writing any asset.
This script never trains, evaluates a checkpoint, or changes results/.
"""
import argparse
import ast
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'report'
NAMES = {'model_a':'Model A','model_b':'Model B','mobilenet_v2':'MobileNetV2','efficientnet_b0':'EfficientNet-B0'}
PREFIX = dict(zip(NAMES, ('A','B','Mobile','Efficient')))

def load(path):
    return json.loads((ROOT/path).read_text())

def assets():
    out = {}
    final = pd.read_csv(ROOT/'results/tables/final_comparison.csv').set_index('model')
    opt = pd.read_csv(ROOT/'results/tables/optimizer_comparison.csv').set_index('optimizer')
    info = load('data/splits/dataset_info.json')
    histories = {n:pd.read_csv(ROOT/f'results/raw/{n}/history.csv') for n in NAMES}
    metas = {n:load(f'results/raw/{n}/run.json') for n in NAMES}
    macros = {}
    def value(key, number, kind='.2f'):
        macros[key] = format(number, kind)
    def table(name, headers, rows, caption):
        out[f'tables/{name}.tex'] = ('% Generated from saved experiment evidence; do not edit values.\n'
            r'\begin{table}[H]\centering\small'+'\n'+r'\caption{'+caption+'}'+r'\label{tab:'+name+'}\n'+
            r'\begin{tabular}{l'+'r'*(len(headers)-1)+'}\n\\toprule\n'+
            ' & '.join(headers)+r' \\ \midrule'+'\n'+
            '\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)+
            '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    for name,row in final.iterrows():
        pre=PREFIX[name]
        for key,col,fmt in [('Params','trainable_parameters',',.0f'),('Accuracy','test_accuracy','.2f'),
                            ('EpochTime','avg_epoch_time_seconds','.3f'),('Latency','cpu_latency_ms_mean','.3f'),
                            ('SizeKiB','model_size_kb','.2f'),('SizeMiB','model_size_mb','.3f')]:
            value(pre+key,row[col]*100 if key=='Accuracy' else row[col],fmt)
        h=histories[name]
        value(pre+'BestEpoch',metas[name]['best_epoch'],'d')
        value(pre+'BestVal',metas[name]['best_val_accuracy']*100)
        value(pre+'FinalTrain',h.train_accuracy.iloc[-1]*100)
        value(pre+'FinalVal',h.val_accuracy.iloc[-1]*100)
        value(pre+'FirstTrainLoss',h.train_loss.iloc[0],'.3f')
        value(pre+'LastTrainLoss',h.train_loss.iloc[-1],'.3f')
        value(pre+'LastValLoss',h.val_loss.iloc[-1],'.3f')
        if name.startswith('model_'):
            value(pre+'WorstDrop',-100*h.val_accuracy.diff().min())
        else:
            value(pre+'HeadVal',h.val_accuracy.iloc[metas[name]['head_epochs']-1]*100)
            value(pre+'FineGain',100*(metas[name]['best_val_accuracy']-h.val_accuracy.iloc[metas[name]['head_epochs']-1]))
    a,b=final.loc['model_a'],final.loc['model_b']
    value('BMargin',100000-int(b.trainable_parameters),',d')
    for k,c in [('ParamReduction','total_parameters'),('StorageReduction','model_size_bytes'),('MacReduction','conv_linear_macs')]:
        value(k,100*(1-b[c]/a[c]))
    value('CustomAccuracyGap',100*(a.test_accuracy-b.test_accuracy))
    value('CustomCorrectGap',round((a.test_accuracy-b.test_accuracy)*info['split_sizes']['test']),'d')
    value('BTrainingIncrease',100*(b.avg_epoch_time_seconds/a.avg_epoch_time_seconds-1))
    value('BCpuIncrease',100*(b.cpu_latency_ms_mean/a.cpu_latency_ms_mean-1))
    for n in ('mobilenet_v2','efficientnet_b0'):
        r=final.loc[n]; pre=PREFIX[n]
        value(pre+'AccuracyGap',100*(r.test_accuracy-b.test_accuracy))
        for k,c in [('ParamSaving','total_parameters'),('StorageSaving','model_size_bytes'),('MacSaving','conv_linear_macs')]:
            value(pre+k,100*(1-b[c]/r[c]))
    value('TransferAccuracyGap',100*(final.loc['mobilenet_v2','test_accuracy']-final.loc['efficientnet_b0','test_accuracy']))
    value('MomentumGain',100*(opt.loc['momentum','best_val_accuracy']-opt.loc['sgd','best_val_accuracy']))
    value('AdamMomentumGap',100*(opt.loc['adam','best_val_accuracy']-opt.loc['momentum','best_val_accuracy']))
    for n,prefix in [('sgd','Sgd'),('momentum','Momentum'),('adam','Adam')]:
        h=pd.read_csv(ROOT/f'results/raw/optimizer_{n}/history.csv')
        value(prefix+'PilotBest',100*opt.loc[n,'best_val_accuracy'])
        value(prefix+'PilotFinalLoss',h.val_loss.iloc[-1],'.3f')
    for n in ('model_a','model_b'):
        frame=pd.read_csv(ROOT/f'results/tables/{n}_architecture.csv',keep_default_na=False)
        for op, suffix in [('Conv2d','ConvParams'),('BatchNorm2d','BnParams'),('Linear','HeadParams')]:
            value(PREFIX[n]+suffix,int(frame.loc[frame.type==op,'parameters'].sum()),',d')
        assert int(frame.parameters.sum())==int(final.loc[n,'total_parameters'])
        rows=[]
        for _,r in frame.iterrows():
            shape=lambda s:'$'+r'\times'.join(map(str,ast.literal_eval(s)))+'$'
            op={'BatchNorm2d':'BN','AdaptiveAvgPool2d':'GAP','MaxPool2d':'MaxPool'}.get(r.type,r.type)
            if r.type=='Conv2d' and int(r.groups)>1: op='DWConv'
            if r.type=='Conv2d' and r.kernel=='(1, 1)': op='PWConv'
            ksp='/'.join(str(r[c]).replace('(','').replace(')','').replace(', ','x') for c in ('kernel','stride','padding'))
            rows.append([r.layer.replace('features','f').replace('classifier','head'),op,
                         shape(r.input_shape),shape(r.output_shape),ksp,str(r.parameters)])
        title=NAMES[n]+' leaf layers, shapes and trainable parameter counts.'
        text=r'\begingroup\scriptsize\setlength{\tabcolsep}{3pt}'+'\n'+r'\begin{longtable}{lllllr}'+'\n'
        text+=r'\caption{'+title+'}'+r'\label{tab:'+n+'_layers}'+r'\\'+'\n'
        header=r'\toprule Layer & Operation & Input (NCHW) & Output & K/S/P & Params \\ \midrule'
        text+=header+r'\endfirsthead'+'\n'+r'\multicolumn{6}{l}{'+NAMES[n]+r' layers (continued)}\\'+header+r'\endhead'+'\n'
        text+='\n'.join(' & '.join(row)+r' \\' for row in rows)
        text+='\n'+r'\midrule\multicolumn{5}{r}{Total trainable parameters} & '+f'{frame.parameters.sum():,}'+r' \\ \bottomrule'+'\n\\end{longtable}\n\\endgroup\n'
        out[f'tables/{n}_layers.tex']=text
    table('dataset',['Class','Total','Train','Validation','Test'],
          [[n,*[info['class_counts'][n][c] for c in ('total','train','val','test')]] for n in info['classes']]+
          [['Total',info['total_images'],*info['split_sizes'].values()]],'EuroSAT class counts and the shared stratified allocation.')
    for file,models in [('custom_accuracy',['model_a','model_b']),('pretrained_accuracy',['mobilenet_v2','efficientnet_b0'])]:
        table(file,['Model',r'Accuracy (\%)',r'Macro P (\%)',r'Macro R (\%)'],
              [[NAMES[n],*[f'{100*final.loc[n,c]:.2f}' for c in ('test_accuracy','macro_precision','macro_recall')]] for n in models],
              'Held-out test metrics for '+('custom' if file.startswith('custom') else 'pretrained')+' models; P is precision and R is recall.')
    table('custom_resources',['Model','Parameters','Size (KiB)','Epoch (s)','MACs (M)'],
          [[NAMES[n],f'{final.loc[n,"total_parameters"]:,.0f}',f'{final.loc[n,"model_size_kb"]:.2f}',f'{final.loc[n,"avg_epoch_time_seconds"]:.3f}',f'{final.loc[n,"conv_linear_macs"]/1e6:.3f}'] for n in ('model_a','model_b')],
          'Custom-model resources. All parameters are trainable; epoch time includes validation.')
    table('pretrained_resources',['Model','Total','Trainable','Size (MiB)','Epoch (s)'],
          [[NAMES[n],f'{final.loc[n,"total_parameters"]:,.0f}',f'{final.loc[n,"trainable_parameters"]:,.0f}',f'{final.loc[n,"model_size_mb"]:.3f}',f'{final.loc[n,"avg_epoch_time_seconds"]:.3f}'] for n in ('mobilenet_v2','efficientnet_b0')],
          'Pretrained-model resources. Trainable counts refer to fine-tuning; timing averages both stages.')
    stage_rows=[]
    for n in ('mobilenet_v2','efficientnet_b0'):
        for stage in ('head','finetune'):
            h=histories[n].query('stage == @stage')
            stage_rows.append([NAMES[n],stage,len(h),f'{h.lr.iloc[0]:g}',f'{metas[n]["stage_trainable_parameters"][stage]:,}',f'{h.epoch_seconds.mean():.3f}'])
    table('transfer_stages',['Model','Stage','Epochs','LR','Trainable','Epoch (s)'],stage_rows,'Transfer-learning stages, learning rates and measured mean epoch times.')
    table('optimizers',['Optimizer','LR','Momentum',r'Best val. (\%)','Epoch',r'Final val. (\%)'],
          [[{'sgd':'SGD','momentum':'SGD + momentum','adam':'Adam'}[n],f'{r.lr:g}','--' if n=='adam' else f'{r.momentum:g}',f'{100*r.best_val_accuracy:.2f}',int(r.best_epoch),f'{100*r.final_val_accuracy:.2f}'] for n,r in opt.iterrows()],
          'Optimizer comparison on Model B; each run uses ten epochs. Adam moment coefficients are specified in the text.')
    pilot_rows=[]
    for n in ('sgd','momentum','adam'):
        h=pd.read_csv(ROOT/f'results/raw/optimizer_{n}/history.csv')
        pilot_rows.append([{'sgd':'SGD','momentum':'SGD + momentum','adam':'Adam'}[n],f'{h.train_loss.iloc[-1]:.3f}',f'{h.val_loss.iloc[-1]:.3f}',f'{100*h.train_accuracy.iloc[-1]:.2f}',f'{h.epoch_seconds.mean():.3f}'])
    table('optimizer_dynamics',['Optimizer','Train loss','Val. loss',r'Train acc. (\%)','Epoch (s)'],pilot_rows,'Final-epoch optimizer losses and training accuracy; mean timing spans all ten epochs.')
    savings=pd.read_csv(ROOT/'results/tables/convolution_savings.csv')
    table('savings',['Channels','Standard','Depthwise','Pointwise',r'Reduction (\%)'],
          [[f'{int(r.cin)} to {int(r.cout)}',int(r.standard_weights),int(r.depthwise_weights),int(r.pointwise_weights),f'{r.reduction_percent:.2f}'] for _,r in savings.iterrows()],
          'Convolution weight counts excluding bias and normalization; reduction compares depthwise plus pointwise with standard convolution.')
    table('compute',['Model','MACs (M)','2xMAC (MFLOP)','CPU mean (ms)','CPU p95 (ms)'],
          [[NAMES[n],f'{r.conv_linear_macs/1e6:.3f}',f'{r.conv_linear_flops_2mac/1e6:.3f}',f'{r.cpu_latency_ms_mean:.3f}',f'{r.cpu_latency_ms_p95:.3f}'] for n,r in final.iterrows()],
          'One 64x64 image: Conv/Linear arithmetic estimates and batch-one host CPU latency.')
    value('APermanentPrecision',pd.read_csv(ROOT/'results/raw/model_a/classification_report.csv',index_col=0).loc['PermanentCrop','precision']*100)
    value('BHighwayRecall',pd.read_csv(ROOT/'results/raw/model_b/classification_report.csv',index_col=0).loc['Highway','recall']*100)
    out['tables/result_values.tex']='% Derived from saved CSV/JSON evidence.\n'+'\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n'
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    for relative,expected in assets().items():
        path=REPORT/relative
        if args.check:
            assert path.read_text(encoding='utf-8')==expected, f'Numeric asset differs from source: {relative}'
        else:
            path.write_text(expected,encoding='utf-8')
    print('Verified' if args.check else 'Generated',len(assets()),'report-only numeric assets.')

if __name__=='__main__':
    main()
