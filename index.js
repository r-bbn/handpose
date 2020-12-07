// Set the backen used by TensorFlow
// See https://www.tensorflow.org/js/guide/platform_environment
const state = {
    backend: 'webgl'
    //backend: 'wasm'
    //backend: 'cpu'
};

// Detect if mobile user
function isMobile() {
    const isAndroid = /Android/i.test(navigator.userAgent);
    const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
    return isAndroid || isiOS;
}

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 500;
const mobile = isMobile();

let videoWidth, videoHeight, canvas, ctx, rafID, model;

// Indices by finger used to draw the keypoints and paths between them
let fingerLookupIndices = {
    thumb: [0, 1, 2, 3, 4],
    indexFinger: [0, 5, 6, 7, 8],
    middleFinger: [0, 9, 10, 11, 12],
    ringFinger: [0, 13, 14, 15, 16],
    pinky: [0, 17, 18, 19, 20]
};

/**
 * Setup the camera and return a video object with
 * an attached stream corresponding to the webcam
 */
async function setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error(
            'Browser API navigator.mediaDevices.getUserMedia not available');
    }

    const video = document.getElementById('video');

    // Get video stream from the webcam
    const stream = await navigator.mediaDevices.getUserMedia({
        'audio': false,
        'video': {
            facingMode: 'user',
            // Only setting the video to a specified size in order to accommodate a
            // point cloud, so on mobile devices accept the default size.
            width: mobile ? undefined : VIDEO_WIDTH,
            height: mobile ? undefined : VIDEO_HEIGHT
        },
    });

    // Attach it to the video
    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

/**
 * Load the video
 */
async function loadVideo() {
    const video = await setupCamera();
    video.play();
    return video;
}

/**
 * Draw the keypoints predicted by the model
 * See the structure of the "prediction" object in the main() function
 * @param keypoints
 */
function drawKeypoints(keypoints) {
    const keypointsArray = keypoints;

    for (let i = 0; i < keypointsArray.length; i++) {
        const y = keypointsArray[i][0];
        const x = keypointsArray[i][1];
        drawPoint(x - 2, y - 2, 3);
    }

    const fingers = Object.keys(fingerLookupIndices);
    for (let i = 0; i < fingers.length; i++) {
        const finger = fingers[i];
        const points = fingerLookupIndices[finger].map(idx => keypoints[idx]);
        drawPath(points, false);
    }
}

/**
 * Draw a point on the canvas
 * @param y coordinate
 * @param x coordinate
 * @param r radius of the circle to fill
 */
function drawPoint(y, x, r) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fill();
}

/**
 * Draw a path between the points of the fingers
 * @param points
 * @param closePath
 */
function drawPath(points, closePath) {
    const region = new Path2D();
    region.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
        const point = points[i];
        region.lineTo(point[0], point[1]);
    }

    if (closePath) {
        region.closePath();
    }
    ctx.stroke(region);
}

/**
 * Detect if the hand in the video is closed
 * @param predictions (the prediction object of the model)
 * @returns hand closed (true/false)
 */
function detectClosedHand(predictions) {

    // Get fingers (not thumb and palmBase, see prediction object in console)

    let nbClosed = 0;

    let keys = Object.keys(predictions[0].annotations);
    keys.forEach(key => {
        if(key != "thumb" && key != "palmBase") {
            nbClosed += detectClosedFinger(predictions[0].annotations[key]);
        }
    })

    // Detect the number of closed fingers

    // If it is >= 2, the hand is considered closed

    return nbClosed >= 2;
}

/**
 * Detect if a finger is closed
 * @param finger
 *  * The structure of a finger is the following :
 * finger {
 *     [
 *         [x0, y0, z0] (1st point coordinates = Bottom)
 *         [x1, y1, z1] (2nd point)
 *         [x2, y2, z2] (3rd point)
 *         [x3, y3, z3] (4th point = Top)
 *     ]
 * }
 * @returns 1 if closed, 0 otherwise
 */
function detectClosedFinger(finger) {

    // Get yTop & yBottom
    let yTop = finger[3][1]
    let yDown = finger[0][1]

    // Return 1 if finger closed, 0 otherwise
    // Warning ! See how the "y" axis is defined !
    if(yTop > yDown) {
        return 1;
    } else {
        return 0;
    }
}

/**
 * Function used to analyse the frames produced by the video stream using Tensorflow
 * @param video (the video stream)
 * @param audio (the audio file that will be play)
 */
const landmarksRealTime = async (video, audio) => {

    async function frameLandmarks() {
        // Draw canvas image using the webcam video stream
        ctx.drawImage(
            video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width,
            canvas.height);

        const predictions = await model.estimateHands(video);

        // If there is a result, forward to analysis
        if (predictions.length > 0) {

            // See the structure of the prediction object here :
            // https://github.com/tensorflow/tfjs-models/tree/master/handpose

            // You can log it to see further details
            console.log("Predictions", predictions);

            const result = predictions[0].landmarks;
            drawKeypoints(result);

            // Detect if the hand is closed, and play audio
            if(detectClosedHand(predictions)) {
                audio.pause();
                console.log('Close');
            } else {
                audio.play();
                console.log('Open');
            }
        }
        // Otherwise if no hands are detected
        else {
            audio.pause();
            console.log('No hand');
        }
        // Launch new analysis for the next frame
        rafID = requestAnimationFrame(frameLandmarks);
    };

    // Launch the function
    frameLandmarks();
};

/**
 * Main Function
 */
async function main() {
    await tf.setBackend(state.backend);
    model = await handpose.load();

    // Load the webcam video stream
    let video;
    try {
        video = await loadVideo();
    } catch (e) {
        // If error, display message
        let info = document.getElementById('info');
        info.textContent = e.message;
        info.style.display = 'block';
        throw e;
    }

    // Set dimensions
    videoWidth = video.videoWidth;
    videoHeight = video.videoHeight;

    // Set dimensions of the canvas where the video is displayed
    canvas = document.getElementById('output');
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    video.width = videoWidth;
    video.height = videoHeight;

    // Set canvas context
    ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, videoWidth, videoHeight);
    ctx.strokeStyle = 'red';
    ctx.fillStyle = 'red';
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);

    // Load Audio
    let audio = new Audio('./resources/music.mp3');
    audio.loop = true;

    landmarksRealTime(video, audio);
}

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

// Execution of the main
main();