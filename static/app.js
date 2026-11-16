const $ = id => document.getElementById(id);
let file = null, previewUrl = null, busy = false;
document.querySelectorAll('[data-panel]').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('[data-panel]').forEach(b => b.setAttribute('aria-selected', String(b === button)));
  ['analyze','evaluate'].forEach(id => $(id).hidden = id !== button.dataset.panel);
}));
function showError(message) { $('error').textContent = message; $('error').hidden = false; }
function selectFile(selected) {
  if (busy) return;
  file = selected;
  $('error').hidden = true; $('result').hidden = true; $('empty').hidden = false;
  $('status').textContent = 'Ready to analyze';
  $('submit').disabled = !file;
  if (!file) return;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $('preview').src = previewUrl; $('preview-wrap').hidden = false;
  $('filename').textContent = file.name;
}
$('image').addEventListener('change', event => selectFile(event.target.files[0]));
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click',async () => {
  if (busy) return;
  try {
    const response=await fetch('/demo/'+button.dataset.example);
    if (!response.ok) throw new Error('Example is unavailable. Choose an image from demo_images.');
    selectFile(new File([await response.blob()],button.dataset.example,{type:'image/png'}));
  } catch(error) {showError(error.message);}
}));
['dragover','dragleave','drop'].forEach(type => $('dropzone').addEventListener(type, event => {
  event.preventDefault(); $('dropzone').classList.toggle('drag', type === 'dragover');
  if (type === 'drop') selectFile(event.dataTransfer.files[0]);
}));
$('upload-form').addEventListener('submit', async event => {
  event.preventDefault(); if (!file || busy) return;
  busy = true; $('image').disabled = true;
  const data = new FormData(); data.append('image', file);
  $('submit').disabled = true; $('status').textContent = 'Analyzing…'; $('error').hidden = true;
  $('result').hidden = true;
  try {
    const response = await fetch('/api/predict', {method:'POST', body:data});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Analysis failed.');
    $('empty').hidden = true; $('result').hidden = false;
    $('prediction').textContent = result.label;
    $('score-details').hidden = result.ai_score === null;
    for (const kind of result.ai_score === null ? [] : ['ai','real']) {
      $(kind+'-value').textContent = (result[kind+'_score']*100).toFixed(1)+'%';
      $(kind+'-bar').style.width = result[kind+'_score']*100+'%';
    }
    $('dimensions').textContent = `Uploaded: ${result.original_size.join(' × ')} px${result.ai_score === null ? ' · Outside photo-classifier scope' : ' · CNN input: 256 × 256 px'}`;
    $('result-note').textContent = result.note;
    $('status').textContent = 'Analysis complete';
  } catch (error) { showError(error.message); $('status').textContent = 'Check image'; }
  finally { busy = false; $('image').disabled = false; $('submit').disabled = false; }
});
async function loadMetrics() {
  try {
    const response = await fetch('/api/metrics'); if (!response.ok) throw new Error('Could not load evaluation.');
    const result = await response.json(), metrics = result.metrics;
    if (!metrics) { $('evaluation-status').textContent = 'Evaluation is not ready yet. Train the CNN to create genuine test results.'; return; }
    const names = {accuracy:'Test accuracy',error_rate:'Error rate',precision:'AI precision',recall:'AI recall',f1:'AI F1 score',roc_auc:'ROC-AUC'};
    for (const [key,label] of Object.entries(names)) {
      const card = document.createElement('div'); card.className = 'metric';
      const title = document.createElement('span'); title.textContent = label;
      const value = document.createElement('strong'); value.textContent = key === 'roc_auc' ? metrics[key].toFixed(3) : (metrics[key]*100).toFixed(1)+'%';
      card.append(title,value); $('metrics-grid').append(card);
    }
    $('evaluation-content').hidden = false;
    $('evaluation-status').textContent = `${metrics.test_size} original test images (reused benchmark) · Best epoch ${metrics.best_epoch} · ${(metrics.trainable_parameters || metrics.parameters).toLocaleString()} trained parameters`;
    const policy=metrics.decision_policy;
    $('policy-details').textContent = policy ? `These binary metrics include every photo at the 0.50 threshold. The website marks scores from 0.35 to 0.65 inconclusive: ${policy.inconclusive_count} of ${metrics.test_size} test photos.` : '';
    if (result.audit) {
      const a=result.audit, p=a.decision_policy;
      $('audit-details').textContent = `${a.test_size} fresh images from a different shard of the same dataset: ${(a.accuracy*100).toFixed(1)}% binary accuracy, ${(a.error_rate*100).toFixed(1)}% error rate, ${(a.recall*100).toFixed(1)}% AI recall. ${p.inconclusive_count} photos receive an inconclusive display result. This is a same-dataset audit, not an independent-source benchmark.`;
    }
    if (result.baseline) $('comparison-details').textContent = `Same 250-image benchmark: original small CNN ${(result.baseline.accuracy*100).toFixed(1)}% accuracy / ${(result.baseline.recall*100).toFixed(1)}% AI recall → transfer CNN ${(metrics.accuracy*100).toFixed(1)}% accuracy / ${(metrics.recall*100).toFixed(1)}% AI recall. The replacement was selected using validation loss, not test scores.`;
    const d = result.dataset;
    $('dataset-details').textContent = d ? `${d.dataset}: ${d.per_class} images per class. Train ${d.splits.train}, validation ${d.splits.validation}, test ${d.splits.test}. ${d.limitations}` : 'See the dataset manifest for source details.';
  } catch (error) { $('evaluation-status').textContent = error.message; }
}
loadMetrics();
