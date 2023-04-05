function initialize() {
  document.getElementById('image-upload').addEventListener('change', (e) => {
	const img = new Image();
	img.src = URL.createObjectURL(e.target.files[0]);
	img.onload = () => {
	  URL.revokeObjectURL(img.src);
	  document.getElementById('image-container').innerHTML = '';
	  document.getElementById('image-container').appendChild(img);
	  identifyBird(img);
	}
  });
}

async function identifyBird(imgElement) {
  const model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  const processedImg = tf.browser.fromPixels(imgElement).toFloat().resizeNearestNeighbor([224, 224]).div(tf.scalar(255)).expandDims();

  const predictions = model.predict(processedImg);
  const top5 = await getTopKClasses(predictions, 5);

  displayResults(top5);
}

async function getTopKClasses(logits, topK) {
  const values = await logits.data();
  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
	valuesAndIndices.push({ value: values[i], index: i });
  }
  valuesAndIndices.sort((a, b) => b.value - a.value);
  const topkValues = valuesAndIndices.slice(0, topK).map(x => x.value);
  const topkIndices = valuesAndIndices.slice(0, topK).map(x => x.index);

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
	topClassesAndProbs.push({
	  label: IMAGENET_CLASSES[topkIndices[i]],
	  probability: topkValues[i]
	});
  }
  return topClassesAndProbs;
}

function displayResults(results) {
  const resultsContainer = document.getElementById('results');
  resultsContainer.innerHTML = '';

  let birdFound = false;
  
  results.forEach((result) => {
	if (result.label.toLowerCase().includes('bird')) {
	  birdFound = true;
	  const resultElement = document.createElement('div');
	  resultElement.innerHTML = `${result.label}: ${(result.probability * 100).toFixed(2)}%`;
	  resultsContainer.appendChild(resultElement);
	}
  });

  if (!birdFound) {
	const noBirdElement = document.createElement('div');
	noBirdElement.innerHTML = 'No bird detected';
	resultsContainer.appendChild(noBirdElement);
  }
}


// Add the IMAGENET_CLASSES object here, which maps class indices to their labels.
// You can find this object in the following
