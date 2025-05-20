
let webCamera = null;
let model = null;
let letterBuffer = [];
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();
const btnLoad = document.getElementById('btnLoad');
const btnSave = document.getElementById('btnSave');
const btnTrain = document.getElementById('btnTrain');
let modelTraining = false;


const classes = { 1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 8: "H", 9: "I", 10: "J", 11: "K", 12: "L", 13: "M", 14: "N", 15: "O", 16: "P", 17: "Q", 18: "R", 19: "S", 20: "T", 21: "U", 22: "V", 23: "W", 24: "X", 25: "Y", 26: "Z" };


let handposeModel = null;


async function loadHandposeModel() {
  handposeModel = await handpose.load();
  console.log("Handpose model loaded");
}


function drawHand(handPredictions, ctx) {
  
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  for (let i = 0; i < handPredictions.length; i++) {
    const keypoints = handPredictions[i].landmarks;

    for (let j = 0; j < keypoints.length; j++) {
      const [x, y, z] = keypoints[j];
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 3 * Math.PI);
      ctx.fillStyle = "red";
      ctx.fill();
    }

    const fingers = handPredictions[i].annotations;
    for (let finger in fingers) {
      const points = fingers[finger];
      ctx.beginPath();
      ctx.moveTo(points[0][0], points[0][1]);

      for (let j = 1; j < points.length; j++) {
        ctx.lineTo(points[j][0], points[j][1]);
      }

      ctx.strokeStyle = "blue";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
}


const clearWord = () => {
  letterBuffer = [];
  processWord();
};

const addImgExample = async (classId) => {
  if (modelTraining) {
    const img = await webCamera.capture();
    const activation = model.infer(img, "conversation predictions");
    classifier.addExample(activation, classId);
    img.dispose();
  } else {
    swal("Oops, the model is not in training mode!", " Click on 'Train' to activate training mode.", "error");
  }
};


const processWord = () => {
  const word = letterBuffer.join('');
  document.getElementById("result2").innerHTML = `
      <p><b>Word:</b> ${word}</p>
  `;
};

const app = async () => {
  try {
    webCamera = await tf.data.webcam(webcamElement);
    model = await mobilenet.load();
    await loadHandposeModel();  

    const interval = setInterval(async () => {
      if (model !== null && classifier.getNumClasses() > 0 && handposeModel !== null) {
        const img = await webCamera.capture();

        const handPredictions = await handposeModel.estimateHands(img);

        if (handPredictions.length > 0) {
          const canvas = document.getElementById('canvas');
          const ctx = canvas.getContext('2d');

          drawHand(handPredictions, ctx);

          const activation = model.infer(img, "conv_preds");
          const result = await classifier.predictClass(activation);

          letterBuffer.push(classes[result.label]);
          document.getElementById("result").innerHTML = `
          <p><b>Lyrics:</b> ${classes[result.label]}</p>
          <p><b>Probability:</b> ${result.confidences[result.label]}</p>
          `;

          processWord();
        }

        img.dispose();
      } else {
        document.getElementById("result").innerHTML = `
      <h4>Oops, we haven't found a trained model.!</h4>
      <p>Please upload a trained model or train a new one. :)</p>
    `;
      }
    }, 2000); 

    const btnClear = document.getElementById("btnClear");
    btnClear.addEventListener("click", () => {
      clearWord();
    });

    window.addEventListener("beforeunload", () => {
      clearInterval(interval);
    });
  } catch (error) {
    console.error("Error loading the model:", error);
  }
};


function secretMessage() {
  swal({
    title: "Hello!",
    text: "If you are our teacher, pass us with 100 uwu",
    icon: "success",
    buttons: {
      si1: {
        text: "Of course!",
      },
      si2: {
        text: "I accept",
      },
      si3: {
        text: "Just say yes :3",
      },
    },
  });
}

function showInstructions() {
  const modal = document.createElement('div');
  modal.classList.add('modal');

  const instructionsContent = `
        <h2>Instructions for Use</h2>
        <p>1. Click on "Train" to start training a new model..</p>
        <p>2.Once the training has started, signal each letter of the alphabet one by one while clicking on the corresponding buttons for each letter to train the model.</p>
        <p>3.You can view the prediction results on the right side of the screen.</p>
        <p>4. It is necessary to grant access to your camera for the web app to function correctly.</p>
        <p>5. Once you are satisfied with the predictions, save the model.</p>
        <p>6. If you already have a model, I can upload it in 'Load model'.</p>
        <p>7. In case of error, refresh the page or check your internet connectivity. :)</p>
        <p>8. Enjoy Sign Lens! :)</p>
    `;

  modal.innerHTML = instructionsContent;

  document.body.appendChild(modal);

  const closeButton = document.createElement('button');
  closeButton.innerText = 'close';
  closeButton.onclick = function () {
    document.body.removeChild(modal);
  };

  modal.appendChild(closeButton);
}

const buttons = document.getElementsByClassName('alpha-btn');
for (let i = 0; i < buttons.length; i++) {
  buttons[i].addEventListener("click", () => {
    const classId = buttons[i].getAttribute("data-position");
    addImgExample(classId);
  });
}


btnLoad.addEventListener("click", async () => {
  const input = document.createElement("input"); 
  input.type = "file"; 
  input.accept = ".json"; 

  input.onchange = async () => {
    const file = input.files[0];
    if (file) {
      try {
        const jsonContent = await file.text(); 
        const loadedDataset = JSON.parse(jsonContent); 
        const tensorObj = Object.entries(loadedDataset).reduce( 
          (obj, [classId, data]) => {
            obj[classId] = tf.tensor(data.data, data.shape, data.dtype);
            return obj;
          },
          {}
        );

        classifier.setClassifierDataset(tensorObj);
      } catch (error) {
        console.error(
          "Error loading the model from the JSON file:",
          error
        );
      }
      swal("Well done!", "Model loaded successfully!", "success", { buttons: false, timer: 2000, });
    }
  };

  input.click();
});

btnTrain.addEventListener("click", () => {
  if (!model) {
    swal("Oops", "The model has not been loaded, please wait a few seconds.", "error");
    return;
  }

  modelTraining = !modelTraining;
  btnTrain.innerText = "Training...";
  btnTrain.disabled = true;
  btnSave.disabled = false;
  swal("Well done!", "Starting training!", "success", {
    buttons: false,
    timer: 3000,
  });

  document.getElementById("alphabet-container").classList.toggle("alphabet-btn-visible", modelTraining);

});

btnSave.addEventListener("click", async () => {
  if (modelTraining) {
    const dataset = classifier.getClassifierDataset();

    const adjustedDataset = Object.entries(dataset).reduce(
      (obj, [classId, data]) => {
        obj[classId] = {
          data: Array.from(data.dataSync()), 
          shape: data.shape,
          dtype: data.dtype,
        };
        return obj;
      },
      {}
    );

    const jsonDataset = JSON.stringify(adjustedDataset); 

    const blob = new Blob([jsonDataset], { type: "application/json" });

    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "modelo_trained.json";

    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    swal("Well Done!", "Model saved successfully!", "success");
  } else {
    swal("Hey stop!", "There is no trained model. First, train one and then save it.", "error");
  }


  modelTraining = false;
  btnTrain.innerText = "train";
  btnTrain.disabled = false;
  btnSave.disabled = true;
});


app();