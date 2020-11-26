const state = {
    backend: 'webgl'
};

function isMobile() {
    const isAndroid = /Android/i.test(navigator.userAgent);
    const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
    return isAndroid || isiOS;
}

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 500;
const mobile = isMobile();

let videoWidth, videoHeight, canvas, ctx, rafID, fingerLookupIndices = {
    thumb: [0, 1, 2, 3, 4],
    indexFinger: [0, 5, 6, 7, 8],
    middleFinger: [0, 9, 10, 11, 12],
    ringFinger: [0, 13, 14, 15, 16],
    pinky: [0, 17, 18, 19, 20]
};

let model;

/**
 * Load video
 * @returns {Promise<unknown>}
 */
async function setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error(
            'Browser API navigator.mediaDevices.getUserMedia not available');
    }

    const video = document.getElementById('video');
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
    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

async function loadVideo() {
    const video = await setupCamera();
    video.play();
    return video;
}

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

function drawPoint(y, x, r) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fill();
}

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

function detect_close_hands(predictions) {

    let indexFinger = predictions[0].annotations.indexFinger;
    let ringFinger = predictions[0].annotations.indexFinger;
    let pinky = predictions[0].annotations.pinky;
    let middleFinger = predictions[0].annotations.middleFinger;

    let nbClosedFingers = detectClosedFinger(indexFinger) + detectClosedFinger(ringFinger) + detectClosedFinger(pinky) + detectClosedFinger(middleFinger);

    return nbClosedFingers >= 2;
}

function detectClosedFinger(finger) {
    let yTop = finger[3][1]
    let yDown = finger[0][1]

    // 'y' croit lorsqu'on va vers le bas
    if(yTop > yDown) {
        return 1;
    }
    else {
        return 0;
    }
}

function drawExtremeFingerPoint(finger) {
    let x = finger[3][0]
    let y = finger[3][1]

    drawPoint(y-2, x-2, 3)

    x = finger[1][0]
    y = finger[1][1]

    drawPoint(y-2, x-2, 3)
}

async function main() {
    await tf.setBackend(state.backend);
    model = await handpose.load();
    let video;

    try {
        video = await loadVideo();
    } catch (e) {
        let info = document.getElementById('info');
        info.textContent = e.message;
        info.style.display = 'block';
        throw e;
    }

    //setupDatGui();

    videoWidth = video.videoWidth;
    videoHeight = video.videoHeight;

    canvas = document.getElementById('output');
    canvas.width = videoWidth;
    canvas.height = videoHeight;
    video.width = videoWidth;
    video.height = videoHeight;

    ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, videoWidth, videoHeight);
    ctx.strokeStyle = 'red';
    ctx.fillStyle = 'red';

    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);

    // // These anchor points allow the hand pointcloud to resize according to its
    // // position in the input.
    // ANCHOR_POINTS = [
    //     [0, 0, 0], [0, -VIDEO_HEIGHT, 0], [-VIDEO_WIDTH, 0, 0],
    //     [-VIDEO_WIDTH, -VIDEO_HEIGHT, 0]
    // ];
    //
    // if (renderPointcloud) {
    //     document.querySelector('#scatter-gl-container').style =
    //         `width: ${VIDEO_WIDTH}px; height: ${VIDEO_HEIGHT}px;`;
    //
    //     scatterGL = new ScatterGL(
    //         document.querySelector('#scatter-gl-container'),
    //         {'rotateOnStart': false, 'selectEnabled': false});
    // }
    //

    // Load Audio
    let audio = new Audio('./resources/music.mp3');
    audio.loop = true;

    // Launch frame analysis
    landmarksRealTime(video, audio);
}

const landmarksRealTime = async (video, audio) => {
    async function frameLandmarks() {
        // stats.begin();
        ctx.drawImage(
            video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width,
            canvas.height);
        const predictions = await model.estimateHands(video);

        if (predictions.length > 0) {
            const result = predictions[0].landmarks;
            drawKeypoints(result, predictions[0].annotations);

            // Detect if the hand is closed..

            if(detect_close_hands(predictions)) {
                audio.pause();
                console.log('Close');
            } else {
                audio.play();
                console.log('Open')
            }

            // if (renderPointcloud === true && scatterGL != null) {
            //     const pointsData = result.map(point => {
            //         return [-point[0], -point[1], -point[2]];
            //     });
            //
            //     const dataset =
            //         new ScatterGL.Dataset([...pointsData, ...ANCHOR_POINTS]);
            //
            //     if (!scatterGLHasInitialized) {
            //         scatterGL.render(dataset);
            //
            //         const fingers = Object.keys(fingerLookupIndices);
            //
            //         scatterGL.setSequences(
            //             fingers.map(finger => ({indices: fingerLookupIndices[finger]})));
            //         scatterGL.setPointColorer((index) => {
            //             if (index < pointsData.length) {
            //                 return 'steelblue';
            //             }
            //             return 'white';  // Hide.
            //         });
            //     } else {
            //         scatterGL.updateDataset(dataset);
            //     }
            //     scatterGLHasInitialized = true;
            // }
        }
        else {
            // Si on ne détecte pas de mains
            audio.pause();
            console.log('No hand')
        }
        // stats.end();
        rafID = requestAnimationFrame(frameLandmarks);
    };

    // Appel en boucle
    frameLandmarks();
};

navigator.getUserMedia = navigator.getUserMedia ||
    navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

main();