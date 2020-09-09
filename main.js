

// fixed parameters
const GRID_WIDTH = 24;
const GRID_CHANNELS = 9;
const FP_SIZE = 2048;


function get(x) {
    return fetch(x).then(x=>x.json());
}

function load_model() {
    return tf.loadGraphModel('/model/voxel_web/model.json');
}

function run_inference(model, grid, smiles, fingerprints) {
    // build tensor from flattened grid
    var tensor = tf.tensor(grid, [1,GRID_CHANNELS,GRID_WIDTH,GRID_WIDTH,GRID_WIDTH]);
    var pred = model.predict(tensor);

    var pred_b = pred.broadcastTo([smiles.length, FP_SIZE]);

    // cosine similarity
    // (a dot b) / (|a| * |b|)
    var dot = tf.sum(tf.mul(fingerprints, pred_b), 1);
    var n1 = tf.norm(fingerprints, 2, 1);
    var n2 = tf.norm(pred_b, 2, 1);
    var d = tf.maximum(tf.mul(n1, n2), 1e-6);
    var dist = tf.div(dot, d).arraySync();

    // join smiles with distance
    var scores = [];
    for (var i = 0; i < smiles.length; ++i) {
        scores.push([smiles[i], dist[i]])
    }

    // sort predictions
    scores.sort((a,b) => b[1] - a[1]);

    return scores;
}

async function main() {

    let info = await get('/info.json');

    // aggregate fingerprints into a single tensor for vector math
    let fp = await get('/fingerprints.json');
    var smiles = Object.keys(fp);
    var fpdat = [];
    for (var k in smiles) {
        fpdat.push(fp[smiles[k]]);
    }
    var fingerprints = tf.tensor(fpdat);

    // load model
    const model = await load_model();

    for (var i = 0; i < info.length; ++i) {
        var grid = await get('/out/grid_' + i + '.json');
        
        var scores = run_inference(model, grid, smiles, fingerprints);

        s = '';
        s += '<div>Orig: <b>' + info[i]['orig'] + '</b></div>';
        for (var j = 0; j < 10; ++j) {
            s += '<div>' + scores[j][0] + ' (' + scores[j][1] + ')</div>';
        }
        s += '<br/>'

        document.getElementById('pred').innerHTML += s;
    }
}

main();
