/* The trained model runs entirely on this device. No upload or inference API. */
let modelSessionPromise = null;
async function loadModel() {
  if (!modelSessionPromise) {
    ort.env.wasm.numThreads = 1; // Works without cross-origin-isolation headers.
    ort.env.wasm.wasmPaths = new URL('./vendor/', document.baseURI).href;
    modelSessionPromise = ort.InferenceSession.create(new URL('./model.onnx', document.baseURI).href,
      {executionProviders:['wasm'], graphOptimizationLevel:'all'}).catch(error => {
        modelSessionPromise = null;
        throw new Error('Could not load the CNN. Check your connection and try again.');
      });
  }
  return modelSessionPromise;
}
async function predictInBrowser(file) {
  if (file.size > 12 * 1024 * 1024) throw new Error('Maximum image size is 12 MB.');
  if (!['image/jpeg','image/png','image/webp'].includes(file.type)) throw new Error('Choose a JPEG, PNG or WebP image.');
  let bitmap;
  try { bitmap = await createImageBitmap(file, {imageOrientation:'from-image'}); }
  catch { throw new Error('This image could not be decoded. Try another file.'); }
  const width=bitmap.width, height=bitmap.height;
  if (Math.min(width,height)<256) {bitmap.close();throw new Error('Image width and height must both be at least 256 pixels.');}
  if (width*height>25000000) {bitmap.close();throw new Error('Please use an image with at most 25 million pixels.');}
  const canvas=document.createElement('canvas');canvas.width=canvas.height=256;
  const context=canvas.getContext('2d',{willReadFrequently:true});
  context.imageSmoothingEnabled=true;context.imageSmoothingQuality='high';
  const side=Math.min(width,height);
  context.drawImage(bitmap,(width-side)/2,(height-side)/2,side,side,0,0,256,256);bitmap.close();
  const rgba=context.getImageData(0,0,256,256).data;
  const pixels=256*256, tensor=new Float32Array(3*pixels);
  for(let i=0;i<pixels;i++) {
    tensor[i]=rgba[i*4]/255;tensor[pixels+i]=rgba[i*4+1]/255;tensor[2*pixels+i]=rgba[i*4+2]/255;
  }
  document.getElementById('status').textContent='Running CNN on your device…';
  const session=await loadModel();
  const input=new ort.Tensor('float32',tensor,[1,3,256,256]);
  const output=await session.run({image:input});
  const score=1/(1+Math.exp(-Number(output.logit.data[0])));
  input.dispose();output.logit.dispose();
  const uncertain=Math.abs(score-0.5)<=0.15;
  return {
    label:uncertain?'Inconclusive':score>=0.5?'Likely AI-generated':'Likely real photograph',
    binary_label:score>=0.5?'AI-generated':'Real',ai_score:score,real_score:1-score,
    original_size:[width,height],input_size:[256,256],
    note:'Scores from 35% to 65% are inconclusive. This is a model estimate, not proof of origin. Images are processed in your browser and are not uploaded.'
  };
}
