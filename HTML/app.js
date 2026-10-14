/**
 * Simulador Interactivo de Probabilidad Condicional de Victoria en Tenis Profesional
 * Versión 9 — Recreación optimizada, modular y simplificada.
 * Inferencia cliente del modelo XGBoost (58 características One-Hot, t, t-1, t-2).
 */

// ==========================================================================
// 1. MOTOR DE INFERENCIA XGBOOST (58 VARIABLES)
// ==========================================================================
class XGBPredictor {
    constructor(model) {
        this.features = model.features;
        this.numFeatures = this.features.length;
        this.baseScore = model.base_score;
        this.baseMargin = Math.log(this.baseScore / (1.0 - this.baseScore));
        this.trees = model.trees;
        this.numTrees = this.trees.length;
        this.featMap = {};
        this.features.forEach((f, i) => { this.featMap[f] = i; });
    }

    predict(features) {
        let margin = this.baseMargin;
        const trees = this.trees;
        const n = trees.length;
        for (let t = 0; t < n; t++) {
            const tr = trees[t];
            const left = tr.left;
            const right = tr.right;
            const splits = tr.splits;
            const conds = tr.conds;
            const weights = tr.weights;
            let node = 0;
            while (left[node] !== -1) {
                node = (features[splits[node]] < conds[node]) ? left[node] : right[node];
            }
            margin += weights[node];
        }
        return 1.0 / (1.0 + Math.exp(-margin));
    }

    encodeFeatures(desc) {
        const x = new Float32Array(this.numFeatures);
        const map = this.featMap;

        x[map['sacador_al_tiro']] = desc.hitter_is_svr ? 1 : 0;

        const pasos = [
            { datos: desc, prefijo: '' },
            { datos: desc.prev, prefijo: 'prev_' },
            { datos: desc.prev2, prefijo: 'anteprev_' }
        ];

        for (const { datos, prefijo } of pasos) {
            if (!datos) {
                x[map[`${prefijo}golpe_ninguno`]] = 1;
                x[map[`${prefijo}direccion_indefinida`]] = 1;
                continue;
            }

            if (datos.depth === '7') x[map[`${prefijo}profundidad_corta`]] = 1;
            else if (datos.depth === '8') x[map[`${prefijo}profundidad_media`]] = 1;
            else if (datos.depth === '9') x[map[`${prefijo}profundidad_profunda`]] = 1;

            if (datos.is_approach) x[map[`${prefijo}subida_a_red`]] = 1;

            const esSaque = ['4', '5', '6'].includes(datos.shot_type);
            const colGolpe = `${prefijo}golpe_${this.mapShotType(datos.shot_type)}`;
            if (map[colGolpe] !== undefined) x[map[colGolpe]] = 1;

            const colDir = `${prefijo}direccion_${this.mapDirection(datos.direction, esSaque)}`;
            if (map[colDir] !== undefined) x[map[colDir]] = 1;
        }

        return x;
    }

    mapShotType(type) {
        const mapa = {
            '4': 'saque_abierto', '5': 'saque_cuerpo', '6': 'saque_t',
            'f': 'derecha', 'b': 'reves', 's': 'slice_reves', 'r': 'slice_derecha',
            'v': 'volea', 'z': 'media_volea', 'o': 'smash',
            't': 'derecha', 'u': 'derecha', 'g': 'derecha'
        };
        return mapa[type] || 'derecha';
    }

    mapDirection(dir, isServe = false) {
        if (isServe) return 'indefinida';
        if (dir === '1') return 'paralelo';
        if (dir === '2') return 'centro';
        if (dir === '3') return 'cruzado';
        return 'indefinida';
    }
}

// ==========================================================================
// 2. ESTADO GLOBAL DE LA APLICACIÓN
// ==========================================================================
let predictor = null;
let shotHistory = [];
let shotIndex = 0;
let serveSide = 'deuce';
let isSecondServe = false;
let isPointFinished = false;
let currentP = 0.64;

let currentMatrixCategory = 'main';
let selectedType = '6';
let selectedDir = '3';
let selectedDepth = '8';
let isApproach = false;
let isWinner = false;
let isAceActive = false;
let isError = false;
let errorMode = 'out';

let customBounceCoords = null;
let serverPos = { x: 42, y: 210 };
let returnerPos = { x: 598, y: 130 };

const COURT = {
    xMin: 50,
    xMax: 590,
    netX: 320,
    serviceLeftX: 175,
    serviceRightX: 465,
    doublesTop: 45,
    doublesBottom: 295,
    singlesTop: 76,
    singlesBottom: 264,
    centerY: 170
};

let evalHistory = [
    {
        step: 0,
        label: '0. Estado previo al saque (0-0) (S)',
        hitter: 'Sacador',
        hitter_is_svr: true,
        p_svr: 0.64,
        p_ret: 0.36,
        delta: 0.0,
        cat: 'Referencia basal',
        coords: null
    }
];

// ==========================================================================
// 3. INICIALIZACIÓN Y TEMAS
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
    try {
        initTheme();
        if (typeof XGB_MODEL !== 'undefined') {
            predictor = new XGBPredictor(XGB_MODEL);
            setStatus(`Modelo XGBoost cargado con éxito (${predictor.numTrees} árboles, ${predictor.numFeatures} variables)`, 'ready');
        }
        initCourt();
        setupCourtInteractions();
        updateUI();
        renderTacticalMatrix();
        renderEvalChart();
        setupKeyboardShortcuts();
    } catch (e) {
        setStatus(`Error de inicialización: ${e.message}`, 'error');
        console.error(e);
    }
});

function initTheme() {
    let savedTheme = 'dark';
    try {
        savedTheme = localStorage.getItem('tennis_app_theme') || 'dark';
    } catch (e) {}
    setTheme(savedTheme, false);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const newTheme = (currentTheme === 'dark') ? 'light' : 'dark';
    setTheme(newTheme, true);
    renderEvalChart();
}

function setTheme(theme, save = true) {
    document.documentElement.setAttribute('data-theme', theme);
    const icon = document.getElementById('theme-toggle-icon');
    const text = document.getElementById('theme-toggle-text');
    if (theme === 'light') {
        if (icon) icon.textContent = '🌙';
        if (text) text.textContent = 'Modo Noche';
    } else {
        if (icon) icon.textContent = '☀️';
        if (text) text.textContent = 'Modo Día';
    }
    if (save) {
        try { localStorage.setItem('tennis_app_theme', theme); } catch (e) {}
    }
}

function setupKeyboardShortcuts() {
    window.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        if (e.key === 'Enter') {
            e.preventDefault();
            registerShot();
        } else if (e.key === 'Escape') {
            e.preventDefault();
            resetAll();
        } else if ((e.ctrlKey && e.key.toLowerCase() === 'z') || e.key === 'Backspace') {
            e.preventDefault();
            undoShot();
        } else if (e.key.toLowerCase() === 'w') {
            e.preventDefault();
            toggleWinner();
        } else if (e.key.toLowerCase() === 'a') {
            e.preventDefault();
            toggleApproach();
        }
    });
}

// ==========================================================================
// 4. CANCHA HORIZONTAL E INTERACCIONES
// ==========================================================================
function initCourt() {
    resetPlayerPositions();
    clearCourtVisuals();
    updateServeActionTexts();
    updateCourtPreview();
}

function resetPlayerPositions() {
    serverPos = (serveSide === 'deuce') ? { x: 42, y: 210 } : { x: 42, y: 130 };
    returnerPos = (serveSide === 'deuce') ? { x: 598, y: 130 } : { x: 598, y: 210 };
}

function clearCourtVisuals() {
    ['court-trajectories', 'court-preview', 'court-landings', 'court-zones-overlay'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.innerHTML = '';
    });
    const badge = document.getElementById('court-shot-badge');
    if (badge) badge.textContent = (shotIndex === 0) ? 'Saque' : `Golpe #${shotIndex}`;
}

function toggleServeSide() {
    if (shotIndex > 0) return;
    serveSide = (serveSide === 'deuce') ? 'ad' : 'deuce';
    const sideText = document.getElementById('court-side-text');
    if (sideText) sideText.textContent = (serveSide === 'deuce') ? 'Deuce' : 'Ventaja (Ad)';

    if (serveSide === 'deuce') {
        selectedType = '6';
        selectedDir = '3';
    } else {
        selectedType = '4';
        selectedDir = '1';
    }

    customBounceCoords = null;
    resetPlayerPositions();
    updateCourtPreview();
    updateServeActionTexts();
    renderTacticalMatrix();
}

function setupCourtInteractions() {
    const svg = document.getElementById('tennis-court-svg');
    if (!svg) return;

    let isDraggingBounce = false;

    function updateBounceFromEvent(e) {
        if (isPointFinished) return;
        const pt = svg.createSVGPoint();
        pt.x = e.clientX;
        pt.y = e.clientY;
        const svgP = pt.matrixTransform(svg.getScreenCTM().inverse());

        let newX = Math.round(Math.max(30, Math.min(610, svgP.x)));
        let newY = Math.round(Math.max(40, Math.min(300, svgP.y)));

        customBounceCoords = { x: newX, y: newY };

        if (shotIndex > 0) {
            const hitterIsSvr = (shotIndex % 2 === 0);
            const fromY = hitterIsSvr ? serverPos.y : returnerPos.y;
            const isHitterTop = (fromY < COURT.centerY);

            if (newY < 135) selectedDir = isHitterTop ? '1' : '3';
            else if (newY > 205) selectedDir = isHitterTop ? '3' : '1';
            else selectedDir = '2';

            if (hitterIsSvr) {
                if (newX <= 465) selectedDepth = '7';
                else if (newX <= 530) selectedDepth = '8';
                else selectedDepth = '9';
            } else {
                if (newX >= 175) selectedDepth = '7';
                else if (newX >= 110) selectedDepth = '8';
                else selectedDepth = '9';
            }

            document.querySelectorAll('#depth-row .btn-depth').forEach(b => {
                b.classList.toggle('selected', b.dataset.depth === selectedDepth);
            });
        } else {
            if (serveSide === 'deuce') {
                if (newY < 110) selectedDir = '1';
                else if (newY < 145) selectedDir = '2';
                else selectedDir = '3';
            } else {
                if (newY > 230) selectedDir = '1';
                else if (newY > 195) selectedDir = '2';
                else selectedDir = '3';
            }
        }

        updateCourtPreview();
        renderTacticalMatrix();
        updateLiveEvaluationPreview();
    }

    svg.addEventListener('pointerdown', (e) => {
        isDraggingBounce = true;
        updateBounceFromEvent(e);
    });
    window.addEventListener('pointermove', (e) => {
        if (!isDraggingBounce) return;
        updateBounceFromEvent(e);
    });
    window.addEventListener('pointerup', () => {
        isDraggingBounce = false;
    });
}

function getShotCoordinates(hitterIsSvr, type, dir, depth, currentShotIdx, isErr = false, errMode = 'out') {
    let fromX, fromY, toX, toY;

    if (currentShotIdx === 0) {
        fromX = serverPos.x;
        fromY = serverPos.y;

        if (customBounceCoords && !isErr) {
            return { fromX, fromY, toX: customBounceCoords.x, toY: customBounceCoords.y };
        }

        if (serveSide === 'deuce') {
            if (dir === '1') { toX = 445; toY = 90; }
            else if (dir === '2') { toX = 395; toY = 125; }
            else { toX = 445; toY = 166; }
        } else {
            if (dir === '1') { toX = 445; toY = 250; }
            else if (dir === '2') { toX = 395; toY = 215; }
            else { toX = 445; toY = 174; }
        }
    } else {
        fromX = hitterIsSvr ? serverPos.x : returnerPos.x;
        fromY = hitterIsSvr ? serverPos.y : returnerPos.y;
        const isHitterTop = (fromY < COURT.centerY);

        if (isErr) {
            if (errMode === 'net') {
                toX = COURT.netX;
                toY = isHitterTop ? 140 : 200;
            } else {
                toX = hitterIsSvr ? (COURT.xMax + 26) : (COURT.xMin - 26);
                toY = (dir === '1') ? (isHitterTop ? 56 : 284) : ((dir === '3') ? (isHitterTop ? 284 : 56) : COURT.centerY);
            }
            return { fromX, fromY, toX, toY };
        }

        if (customBounceCoords) {
            return { fromX, fromY, toX: customBounceCoords.x, toY: customBounceCoords.y };
        }

        toY = (dir === '1') ? (isHitterTop ? 95 : 245) : ((dir === '3') ? (isHitterTop ? 245 : 95) : 170);

        if (!hitterIsSvr) {
            if (['t', 'u'].includes(type)) toX = 260;
            else if (depth === '7') toX = 240;
            else if (depth === '8') toX = 142;
            else if (depth === '9') toX = 80;
            else toX = 142;
        } else {
            if (['t', 'u'].includes(type)) toX = 380;
            else if (depth === '7') toX = 400;
            else if (depth === '8') toX = 498;
            else if (depth === '9') toX = 560;
            else toX = 498;
        }
    }

    return { fromX, fromY, toX, toY };
}

function createCurvedPath(x1, y1, x2, y2, hitterIsSvr) {
    const midX = (x1 + x2) / 2;
    const curvature = hitterIsSvr ? -22 : 22;
    const ctrlY = (y1 + y2) / 2 + curvature;
    return `M ${x1} ${y1} Q ${midX} ${ctrlY} ${x2} ${y2}`;
}

function updateCourtPreview() {
    const previewGroup = document.getElementById('court-preview');
    if (!previewGroup) return;
    previewGroup.innerHTML = '';

    if (isPointFinished) {
        renderCourtZones();
        return;
    }

    const hitterIsSvr = (shotIndex % 2 === 0);
    const coords = getShotCoordinates(hitterIsSvr, selectedType, selectedDir, selectedDepth, shotIndex, isError, errorMode);

    const pathD = createCurvedPath(coords.fromX, coords.fromY, coords.toX, coords.toY, hitterIsSvr);
    const markerId = isError ? 'arrow-fault' : 'arrow-preview';
    const strokeColor = isError ? '#f87171' : '#fbbf24';

    let html = `
        <path d="${pathD}" fill="none" stroke="${strokeColor}" stroke-width="2.5" 
              stroke-dasharray="6,4" marker-end="url(#${markerId})" opacity="0.95"/>
        <g class="bounce-target" transform="translate(${coords.toX}, ${coords.toY})">
            <circle class="bounce-target-outer" cx="0" cy="0" r="10" fill="none" stroke="${strokeColor}" stroke-width="1.8"/>
            <circle cx="0" cy="0" r="4.5" fill="${strokeColor}"/>
        </g>
    `;
    previewGroup.innerHTML = html;
    renderCourtZones();
}

function renderShotOnCourt(shot, index) {
    const trajGroup = document.getElementById('court-trajectories');
    const landGroup = document.getElementById('court-landings');
    if (!trajGroup || !landGroup) return;

    const pathD = createCurvedPath(shot.coords.fromX, shot.coords.fromY, shot.coords.toX, shot.coords.toY, shot.hitter_is_svr);
    const color = shot.isError ? '#f87171' : (shot.hitter_is_svr ? '#34d399' : '#60a5fa');
    const marker = shot.isError ? 'arrow-fault' : (shot.hitter_is_svr ? 'arrow-server' : 'arrow-returner');

    const pathEl = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    pathEl.setAttribute('d', pathD);
    pathEl.setAttribute('fill', 'none');
    pathEl.setAttribute('stroke', color);
    pathEl.setAttribute('stroke-width', '2.5');
    pathEl.setAttribute('marker-end', `url(#${marker})`);
    pathEl.setAttribute('opacity', '0.85');
    trajGroup.appendChild(pathEl);

    const circEl = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circEl.setAttribute('cx', shot.coords.toX);
    circEl.setAttribute('cy', shot.coords.toY);
    circEl.setAttribute('r', '4.5');
    circEl.setAttribute('fill', color);
    circEl.setAttribute('stroke', '#ffffff');
    circEl.setAttribute('stroke-width', '1.2');
    landGroup.appendChild(circEl);

    if (shot.hitter_is_svr) {
        returnerPos = { x: Math.max(50, Math.min(610, shot.coords.toX + 22)), y: shot.coords.toY };
    } else {
        serverPos = { x: Math.max(30, Math.min(590, shot.coords.toX - 22)), y: shot.coords.toY };
    }
}

// ==========================================================================
// 5. ZONAS DELIMITADORAS DE LA CANCHA
// ==========================================================================
function renderCourtZones() {
    const zonesGroup = document.getElementById('court-zones-overlay');
    if (!zonesGroup) return;
    zonesGroup.innerHTML = '';
    if (isPointFinished) return;

    if (shotIndex === 0) {
        renderServeZones(zonesGroup);
    } else {
        renderRallyZones(zonesGroup);
    }
}

function renderServeZones(container) {
    const boxX = 320;
    const boxW = 145;
    const sliceH = 94 / 3;

    let slices = (serveSide === 'deuce') ? [
        { dir: '1', name: 'ABIERTO', y: 76 },
        { dir: '2', name: 'AL CUERPO', y: 76 + sliceH },
        { dir: '3', name: 'A LA T', y: 76 + 2 * sliceH }
    ] : [
        { dir: '3', name: 'A LA T', y: 170 },
        { dir: '2', name: 'AL CUERPO', y: 170 + sliceH },
        { dir: '1', name: 'ABIERTO', y: 170 + 2 * sliceH }
    ];

    let html = '';
    slices.forEach((s, idx) => {
        const isActive = (selectedDir === s.dir);
        const midY = s.y + sliceH / 2;
        html += `
            <g class="court-zone-group" style="cursor: pointer;" onclick="selectServeDir('${s.dir}')">
                <rect class="court-zone-rect ${isActive ? 'active' : ''}" 
                      x="${boxX}" y="${s.y}" width="${boxW}" height="${sliceH}" 
                      fill="${isActive ? 'rgba(251, 191, 36, 0.12)' : 'rgba(255, 255, 255, 0.02)'}" 
                      stroke="${isActive ? 'rgba(251, 191, 36, 0.5)' : 'none'}"/>
                <text x="${boxX + boxW / 2}" y="${midY + 3}" text-anchor="middle" 
                      class="court-zone-label-subtle ${isActive ? 'active-zone-text' : ''}">
                    ${s.name}
                </text>
            </g>
        `;
        if (idx > 0) {
            html += `<line class="court-zone-divider" x1="${boxX}" y1="${s.y}" x2="${boxX + boxW}" y2="${s.y}" 
                           stroke="rgba(255, 255, 255, 0.18)" stroke-dasharray="3,3" stroke-width="1"/>`;
        }
    });
    container.innerHTML = html;
}

function renderRallyZones(container) {
    const hitterIsSvr = (shotIndex % 2 === 0);
    const fromY = hitterIsSvr ? serverPos.y : returnerPos.y;
    const isHitterTop = (fromY < COURT.centerY);

    const xLeft = hitterIsSvr ? 320 : 50;
    const xRight = hitterIsSvr ? 590 : 320;
    const targetW = xRight - xLeft;

    let xShort, xMid, xDeep;
    if (hitterIsSvr) {
        xShort = { start: 320, width: 145 };
        xMid = { start: 465, width: 65 };
        xDeep = { start: 530, width: 60 };
    } else {
        xDeep = { start: 50, width: 60 };
        xMid = { start: 110, width: 65 };
        xShort = { start: 175, width: 145 };
    }

    const yParalelo = isHitterTop ? { start: 76, height: 60 } : { start: 204, height: 60 };
    const yCentro = { start: 136, height: 68 };
    const yCruzado = isHitterTop ? { start: 204, height: 60 } : { start: 76, height: 60 };

    const depths = [
        { id: '7', name: 'CORTA', x: xShort },
        { id: '8', name: 'MEDIA', x: xMid },
        { id: '9', name: 'PROFUNDA', x: xDeep }
    ];

    const directions = [
        { id: '1', name: 'PARALELO', y: yParalelo },
        { id: '2', name: 'CENTRO', y: yCentro },
        { id: '3', name: 'CRUZADO', y: yCruzado }
    ];

    let html = '';
    directions.forEach(d => {
        depths.forEach(dp => {
            const isActive = (selectedDir === d.id && selectedDepth === dp.id);
            html += `
                <g class="court-zone-group" style="cursor: pointer;" onclick="selectRallyZone('${d.id}', '${dp.id}')">
                    <rect class="court-zone-rect ${isActive ? 'active' : ''}" 
                          x="${dp.x.start}" y="${d.y.start}" width="${dp.x.width}" height="${d.y.height}" 
                          fill="${isActive ? 'rgba(251, 191, 36, 0.12)' : 'rgba(255, 255, 255, 0.015)'}" 
                          stroke="${isActive ? 'rgba(251, 191, 36, 0.45)' : 'none'}"/>
                </g>
            `;
        });
    });

    [yCentro.start, yCentro.start + yCentro.height].forEach(yLine => {
        html += `<line x1="${xLeft}" y1="${yLine}" x2="${xRight}" y2="${yLine}" 
                       stroke="rgba(255, 255, 255, 0.12)" stroke-dasharray="3,3" stroke-width="0.8"/>`;
    });

    if (hitterIsSvr) {
        html += `<line x1="530" y1="76" x2="530" y2="264" 
                       stroke="rgba(255, 255, 255, 0.12)" stroke-dasharray="3,3" stroke-width="0.8"/>`;
    } else {
        html += `<line x1="110" y1="76" x2="110" y2="264" 
                       stroke="rgba(255, 255, 255, 0.12)" stroke-dasharray="3,3" stroke-width="0.8"/>`;
    }

    container.innerHTML = html;
}

function selectServeDir(dir) {
    selectedDir = dir;
    selectedType = (dir === '1') ? '4' : ((dir === '2') ? '5' : '6');
    customBounceCoords = null;
    updateCourtPreview();
    renderTacticalMatrix();
    updateLiveEvaluationPreview();
}

function selectRallyZone(dir, depth) {
    selectedDir = dir;
    selectedDepth = depth;
    customBounceCoords = null;
    document.querySelectorAll('#depth-row .btn-depth').forEach(b => {
        b.classList.toggle('selected', b.dataset.depth === depth);
    });
    updateCourtPreview();
    renderTacticalMatrix();
    updateLiveEvaluationPreview();
}

// ==========================================================================
// 6. MATRIZ TÁCTICA DE DECISIÓN
// ==========================================================================
function evaluateShotOption(shotType, dir, depth, approach, hitterIsSvr, prev, prev2) {
    const desc = {
        hitter_is_svr: hitterIsSvr,
        shot_type: shotType,
        direction: dir,
        depth: depth,
        is_approach: approach,
        prev: prev,
        prev2: prev2
    };
    const feat = predictor.encodeFeatures(desc);
    const pPred = Math.round(predictor.predict(feat) * 100) / 100;
    const delta = Math.round((hitterIsSvr ? (pPred - currentP) : (currentP - pPred)) * 100) / 100;
    return { pPred, delta };
}

function createCellHtml(type, dir, name, info, isSelected, isBest) {
    const deltaClass = info.delta > 0.005 ? 'delta-pos' : (info.delta < -0.005 ? 'delta-neg' : 'delta-neu');
    const deltaSign = info.delta >= 0 ? '+' : '';
    const deltaStr = `${deltaSign}${info.delta.toFixed(2)}`;
    return `<td>
        <button class="matrix-cell-btn ${isSelected ? 'cell-selected' : ''}" 
                data-type="${type}" data-dir="${dir}"
                onclick="selectMatrixCell('${type}', '${dir}')"
                ondblclick="confirmMatrixCell('${type}', '${dir}')"
                title="${name} — Doble clic para registrar">
            <div class="cell-eval-line">
                <span class="cell-p-val">${info.pPred.toFixed(2)}</span>
                <span class="cell-delta ${deltaClass}">${deltaStr}</span>
                ${isBest ? '<span class="cell-badge-opt">ÓPTIMO</span>' : ''}
            </div>
        </button>
    </td>`;
}

function renderTacticalMatrix() {
    const container = document.getElementById('tactical-matrix-container');
    if (!container || !predictor) return;

    const catTabs = document.getElementById('matrix-category-tabs');
    const secRow = document.getElementById('matrix-secondary-row');
    const serveActions = document.getElementById('serve-actions');
    const rallyActions = document.getElementById('rally-actions');

    if (isPointFinished) {
        if (serveActions) serveActions.style.display = 'none';
        if (rallyActions) rallyActions.style.display = 'none';
        container.innerHTML = '<div style="text-align:center; padding: 26px; color: var(--text-muted); font-size:13px;">Punto finalizado. Hacé clic en <strong>Reiniciar punto (Esc)</strong> o <strong>Deshacer</strong> para volver a simular.</div>';
        return;
    }

    const isServe = (shotIndex === 0);
    const hitter_is_svr = (shotIndex % 2 === 0);
    const prev = shotHistory.length > 0 ? shotHistory[shotHistory.length - 1] : null;
    const prev2 = shotHistory.length > 1 ? shotHistory[shotHistory.length - 2] : null;

    if (isServe) {
        if (catTabs) catTabs.style.display = 'none';
        if (secRow) secRow.style.display = 'none';
        if (serveActions) serveActions.style.display = 'flex';
        if (rallyActions) rallyActions.style.display = 'none';

        const serveOptions = [
            { type: '4', dir: '1', name: 'Saque Abierto', colLabel: 'Abierto' },
            { type: '5', dir: '2', name: 'Saque al Cuerpo', colLabel: 'Al cuerpo' },
            { type: '6', dir: '3', name: 'Saque a la T', colLabel: 'A la T' }
        ];

        let cellCalculations = {};
        let bestDelta = -999;
        let bestKey = null;

        serveOptions.forEach(opt => {
            const { pPred, delta } = evaluateShotOption(opt.type, opt.dir, '0', false, true, null, null);
            const key = `${opt.type}_${opt.dir}`;
            cellCalculations[key] = { pPred, delta, name: opt.name };
            if (delta > bestDelta) {
                bestDelta = delta;
                bestKey = key;
            }
        });

        const curKey = `${selectedType}_${selectedDir}`;
        if (!cellCalculations[curKey]) {
            const defKey = (serveSide === 'ad') ? '4_1' : (bestKey || '6_3');
            const [dT, dD] = defKey.split('_');
            selectedType = dT;
            selectedDir = dD;
        }

        let tableHtml = '<table class="matrix-table"><thead><tr>';
        tableHtml += `<th class="matrix-th-corner">${isSecondServe ? '2do Saque' : '1er Saque'}</th>`;
        serveOptions.forEach(opt => {
            tableHtml += `<th class="matrix-th-col">${opt.colLabel}</th>`;
        });
        tableHtml += '</tr></thead><tbody><tr>';
        tableHtml += `<td class="matrix-row-header"><span>Colocación</span></td>`;

        serveOptions.forEach(opt => {
            const key = `${opt.type}_${opt.dir}`;
            const info = cellCalculations[key];
            const isSel = (selectedType === opt.type && selectedDir === opt.dir);
            const isBest = (key === bestKey);
            tableHtml += createCellHtml(opt.type, opt.dir, opt.name, info, isSel, isBest);
        });

        tableHtml += '</tr></tbody></table>';
        container.innerHTML = tableHtml;

        updateCourtPreview();
        updateLiveEvaluationPreview();
        return;
    }

    // PELOTEO
    if (catTabs) catTabs.style.display = 'flex';
    if (secRow) secRow.style.display = 'flex';
    if (serveActions) serveActions.style.display = 'none';
    if (rallyActions) rallyActions.style.display = 'flex';

    let rows = (currentMatrixCategory === 'specials') ? [
        { type: 'v', name: 'Volea FH' },
        { type: 'z', name: 'Volea BH' },
        { type: 'o', name: 'Smash' },
        { type: 't', name: 'Drop FH' },
        { type: 'u', name: 'Drop BH' },
        { type: 'g', name: 'Globo' }
    ] : [
        { type: 'f', name: 'Derecha (FH)' },
        { type: 'b', name: 'Revés (BH)' },
        { type: 's', name: 'Slice Revés (BH)' },
        { type: 'r', name: 'Slice Derecha (FH)' }
    ];

    const cols = [
        { dir: '1', label: 'Paralelo' },
        { dir: '2', label: 'Centro' },
        { dir: '3', label: 'Cruzado' }
    ];

    let cellCalculations = {};
    let bestDelta = -999;
    let bestKey = null;

    rows.forEach(r => {
        cols.forEach(c => {
            const { pPred, delta } = evaluateShotOption(r.type, c.dir, selectedDepth || '8', isApproach, hitter_is_svr, prev, prev2);
            const key = `${r.type}_${c.dir}`;
            cellCalculations[key] = { pPred, delta, name: r.name };
            if (delta > bestDelta) {
                bestDelta = delta;
                bestKey = key;
            }
        });
    });

    if (!selectedType || !selectedDir || !cellCalculations[`${selectedType}_${selectedDir}`]) {
        const defKey = bestKey || `${rows[0].type}_${cols[0].dir}`;
        const [dT, dD] = defKey.split('_');
        selectedType = dT;
        selectedDir = dD;
    }

    let tableHtml = '<table class="matrix-table"><thead><tr>';
    tableHtml += `<th class="matrix-th-corner">Golpe \\ Dirección</th>`;
    cols.forEach(c => { tableHtml += `<th class="matrix-th-col">${c.label}</th>`; });
    tableHtml += '</tr></thead><tbody>';

    rows.forEach(r => {
        tableHtml += `<tr><td class="matrix-row-header"><span>${r.name}</span></td>`;
        cols.forEach(c => {
            const key = `${r.type}_${c.dir}`;
            const info = cellCalculations[key];
            const isSel = (selectedType === r.type && selectedDir === c.dir);
            const isBest = (key === bestKey);
            tableHtml += createCellHtml(r.type, c.dir, `${r.name} ${c.label}`, info, isSel, isBest);
        });
        tableHtml += '</tr>';
    });

    tableHtml += '</tbody></table>';
    container.innerHTML = tableHtml;

    updateCourtPreview();
    updateLiveEvaluationPreview();
}

function selectMatrixCell(type, dir) {
    selectedType = type;
    selectedDir = dir;
    customBounceCoords = null;
    isError = false;
    isWinner = false;
    isAceActive = false;
    updateCourtPreview();
    renderTacticalMatrix();
    updateLiveEvaluationPreview();
}

function confirmMatrixCell(type, dir) {
    selectMatrixCell(type, dir);
    registerShot();
}

function switchMatrixCategory(cat) {
    currentMatrixCategory = cat;
    document.getElementById('tab-main-shots').classList.toggle('active', cat === 'main');
    document.getElementById('tab-special-shots').classList.toggle('active', cat === 'specials');
    selectedType = (cat === 'specials') ? 'v' : 'f';
    selectedDir = '3';
    customBounceCoords = null;
    renderTacticalMatrix();
}

function selectDepth(btn) {
    document.querySelectorAll('#depth-row .btn-depth').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    selectedDepth = btn.dataset.depth;
    customBounceCoords = null;
    updateCourtPreview();
    renderTacticalMatrix();
    updateLiveEvaluationPreview();
}

function toggleApproach() {
    isApproach = !isApproach;
    const btn = document.getElementById('btn-approach');
    if (btn) btn.classList.toggle('active', isApproach);
    const ind = document.getElementById('approach-indicator');
    if (ind) ind.textContent = isApproach ? '●' : '○';
    renderTacticalMatrix();
    updateLiveEvaluationPreview();
}

function toggleWinner() {
    isWinner = !isWinner;
    const btn = document.getElementById('btn-winner');
    if (btn) btn.classList.toggle('active', isWinner);
    const ind = document.getElementById('winner-indicator');
    if (ind) ind.textContent = isWinner ? '●' : '○';
    updateLiveEvaluationPreview();
}

// ==========================================================================
// 7. ACCIONES DE TIRO (REGISTRO, FALTAS, ERRORES, UNDO, RESET)
// ==========================================================================
function registerShot() {
    if (isPointFinished || !predictor) return;

    const hitterIsSvr = (shotIndex % 2 === 0);
    const coords = getShotCoordinates(hitterIsSvr, selectedType, selectedDir, selectedDepth, shotIndex, isError, errorMode);

    let nextP = currentP;
    let labelDesc = '';

    if (isError) {
        nextP = hitterIsSvr ? 0.00 : 1.00;
        isPointFinished = true;
        labelDesc = 'Error no forzado';
    } else if (isWinner || isAceActive) {
        nextP = hitterIsSvr ? 1.00 : 0.00;
        isPointFinished = true;
        labelDesc = isAceActive ? 'Ace' : 'Tiro ganador';
    } else {
        const desc = {
            hitter_is_svr: hitterIsSvr,
            shot_type: selectedType,
            direction: selectedDir,
            depth: (shotIndex === 0) ? '0' : selectedDepth,
            is_approach: isApproach,
            prev: shotHistory.length > 0 ? shotHistory[shotHistory.length - 1] : null,
            prev2: shotHistory.length > 1 ? shotHistory[shotHistory.length - 2] : null
        };
        const feat = predictor.encodeFeatures(desc);
        nextP = Math.round(predictor.predict(feat) * 100) / 100;
        labelDesc = formatShotLabel(selectedType, selectedDir, selectedDepth, isApproach);
    }

    const delta = Math.round((nextP - currentP) * 100) / 100;
    currentP = nextP;

    const shotRecord = {
        shot_type: selectedType,
        direction: selectedDir,
        depth: (shotIndex === 0) ? '0' : selectedDepth,
        is_approach: isApproach,
        hitter_is_svr: hitterIsSvr,
        coords: coords,
        isError: isError,
        isWinner: isWinner || isAceActive
    };
    shotHistory.push(shotRecord);

    const stepNum = evalHistory.length;
    const playerTag = hitterIsSvr ? '(S)' : '(R)';
    evalHistory.push({
        step: stepNum,
        label: `${stepNum}. ${labelDesc} ${playerTag}`,
        hitter: hitterIsSvr ? 'Sacador' : 'Restador',
        hitter_is_svr: hitterIsSvr,
        p_svr: nextP,
        p_ret: Math.round((1.0 - nextP) * 100) / 100,
        delta: delta,
        cat: evaluateTacticalCategory(delta, hitterIsSvr),
        coords: coords,
        shotDesc: `${labelDesc} ${playerTag}`
    });

    renderShotOnCourt(shotRecord, shotIndex);
    shotIndex++;

    customBounceCoords = null;
    isApproach = false;
    isWinner = false;
    isAceActive = false;
    isError = false;

    const appBtn = document.getElementById('btn-approach');
    if (appBtn) appBtn.classList.remove('active');
    const appInd = document.getElementById('approach-indicator');
    if (appInd) appInd.textContent = '○';

    const winBtn = document.getElementById('btn-winner');
    if (winBtn) winBtn.classList.remove('active');
    const winInd = document.getElementById('winner-indicator');
    if (winInd) winInd.textContent = '○';

    const aceBtn = document.getElementById('btn-serve-ace');
    if (aceBtn) aceBtn.classList.remove('active');
    const aceInd = document.getElementById('serve-ace-indicator');
    if (aceInd) aceInd.textContent = '○';

    if (shotIndex === 1) {
        selectedType = 'f';
        selectedDir = '3';
        selectedDepth = '8';
    }

    updateUI();
    renderTacticalMatrix();
    updateCourtPreview();
    renderEvalChart();

    const winnerName = (currentP >= 0.5) ? 'Sacador' : 'Restador';
    const statusMsg = isPointFinished 
        ? `¡Punto finalizado! Ganador: ${winnerName}. Probabilidad final: ${(currentP * 100).toFixed(0)}%`
        : `Tiro #${shotIndex} registrado: ${labelDesc}. Probabilidad Sacador: ${(currentP * 100).toFixed(0)}%`;
    setStatus(statusMsg, isPointFinished ? 'ready' : '');
}

function handleServeFault() {
    if (shotIndex !== 0 || isPointFinished) return;

    if (!isSecondServe) {
        isSecondServe = true;
        currentP = 0.51;
        evalHistory.push({
            step: evalHistory.length,
            label: '1. 1er Saque falta en red (S)',
            hitter: 'Sacador',
            hitter_is_svr: true,
            p_svr: 0.51,
            p_ret: 0.49,
            delta: -0.13,
            cat: 'Imprecisión táctica',
            coords: null,
            shotDesc: '1er Saque falta en red (S)'
        });
        selectedType = '6';
        selectedDir = '3';
        customBounceCoords = null;

        updateUI();
        renderTacticalMatrix();
        updateCourtPreview();
        renderEvalChart();
        setStatus('Falta del 1er saque. Servidor pasa a 2do saque (Probabilidad: 51%).', '');
    } else {
        isPointFinished = true;
        currentP = 0.00;
        evalHistory.push({
            step: evalHistory.length,
            label: '2. Doble falta (Punto Restador)',
            hitter: 'Sacador',
            hitter_is_svr: true,
            p_svr: 0.00,
            p_ret: 1.00,
            delta: -0.51,
            cat: 'Error grave (Doble falta)',
            coords: null,
            shotDesc: 'Doble falta (S)'
        });

        updateUI();
        renderTacticalMatrix();
        clearCourtVisuals();
        renderEvalChart();
        setStatus('¡Doble falta! Punto finalizado para el Restador.', 'error');
    }
}

function toggleServeAce() {
    isAceActive = !isAceActive;
    const btn = document.getElementById('btn-serve-ace');
    if (btn) btn.classList.toggle('active', isAceActive);
    const ind = document.getElementById('serve-ace-indicator');
    if (ind) ind.textContent = isAceActive ? '●' : '○';
    updateLiveEvaluationPreview();
}

function executeRallyError(type) {
    if (isPointFinished) return;
    isError = true;
    errorMode = (type === 'unforced') ? 'out' : 'net';
    registerShot();
}

function undoShot() {
    if (shotHistory.length === 0) return;

    shotHistory.pop();
    shotIndex = shotHistory.length;
    isPointFinished = false;

    if (evalHistory.length > 1) {
        evalHistory.pop();
        currentP = evalHistory[evalHistory.length - 1].p_svr;
    } else {
        currentP = 0.64;
    }

    if (shotIndex === 0) {
        isSecondServe = false;
        selectedType = (serveSide === 'deuce') ? '6' : '4';
        selectedDir = (serveSide === 'deuce') ? '3' : '1';
    }

    clearCourtVisuals();
    resetPlayerPositions();
    shotHistory.forEach((shot, idx) => {
        renderShotOnCourt(shot, idx);
    });

    customBounceCoords = null;
    isApproach = false;
    isWinner = false;
    isAceActive = false;
    isError = false;

    updateUI();
    renderTacticalMatrix();
    updateCourtPreview();
    renderEvalChart();
    setStatus(`Último tiro deshecho. En turno: Tiro #${shotIndex}.`, '');
}

function resetAll() {
    shotHistory = [];
    shotIndex = 0;
    isSecondServe = false;
    isPointFinished = false;
    currentP = 0.64;

    selectedType = (serveSide === 'deuce') ? '6' : '4';
    selectedDir = (serveSide === 'deuce') ? '3' : '1';
    selectedDepth = '8';
    isApproach = false;
    isWinner = false;
    isAceActive = false;
    isError = false;
    customBounceCoords = null;

    evalHistory = [
        {
            step: 0,
            label: '0. Estado previo al saque (0-0) (S)',
            hitter: 'Sacador',
            hitter_is_svr: true,
            p_svr: 0.64,
            p_ret: 0.36,
            delta: 0.0,
            cat: 'Referencia basal',
            coords: null
        }
    ];

    clearCourtVisuals();
    resetPlayerPositions();
    updateUI();
    renderTacticalMatrix();
    updateCourtPreview();
    renderEvalChart();
    setStatus('Punto reiniciado al estado inicial (0-0).', 'ready');
}

// ==========================================================================
// 8. ACTUALIZACIÓN VISUAL (EVAL BAR, CHART, STATUS)
// ==========================================================================
function updateUI() {
    updateBar(currentP);

    const isServer = (shotIndex % 2 === 0);
    const badge = document.getElementById('hitter-badge');
    if (badge) {
        badge.className = `hitter-badge ${isServer ? 'server-hitting' : 'returner-hitting'}`;
        const hitterText = badge.querySelector('.hitter-text');
        if (hitterText) hitterText.textContent = isServer ? 'Turno del Sacador' : 'Turno del Restador';
        const shotNum = document.getElementById('shot-num');
        if (shotNum) shotNum.textContent = `Tiro #${shotIndex}`;
    }

    updateServeActionTexts();
}

function updateBar(p) {
    const pServer = Math.max(0, Math.min(1, p));
    const pServerPct = Math.round(pServer * 100);
    const pReturnerPct = 100 - pServerPct;

    const fill = document.getElementById('eval-bar-fill');
    if (fill) fill.style.width = `${pServerPct}%`;

    const sPct = document.getElementById('server-pct');
    if (sPct) sPct.textContent = `${pServerPct}%`;

    const rPct = document.getElementById('returner-pct');
    if (rPct) rPct.textContent = `${pReturnerPct}%`;

    const val = document.getElementById('bar-value');
    if (val) val.textContent = pServer.toFixed(2);
}

function updateLiveEvaluationPreview() {
    if (!predictor || isPointFinished) return;

    if (isError) {
        updateBar((shotIndex % 2 === 0) ? 0.0 : 1.0);
        return;
    }
    if (isWinner || isAceActive) {
        updateBar((shotIndex % 2 === 0) ? 1.0 : 0.0);
        return;
    }

    const desc = {
        hitter_is_svr: (shotIndex % 2 === 0),
        shot_type: selectedType,
        direction: selectedDir,
        depth: (shotIndex === 0) ? '0' : selectedDepth,
        is_approach: isApproach,
        prev: shotHistory.length > 0 ? shotHistory[shotHistory.length - 1] : null,
        prev2: shotHistory.length > 1 ? shotHistory[shotHistory.length - 2] : null
    };

    const feat = predictor.encodeFeatures(desc);
    const p = predictor.predict(feat);
    updateBar(p);
}

function updateServeActionTexts() {
    const faultBtnText = document.getElementById('btn-fault-text');
    if (faultBtnText) {
        faultBtnText.textContent = isSecondServe ? 'Doble falta (Punto Restador)' : 'Falta 1er saque';
    }
}

function renderEvalChart() {
    const svg = document.getElementById('eval-chart-svg');
    if (!svg) return;

    const totalSteps = evalHistory.length;
    if (totalSteps <= 1) {
        svg.innerHTML = '<text x="460" y="165" text-anchor="middle" fill="var(--text-muted)" font-size="12">Esperando tiros del punto para graficar la evolución...</text>';
        return;
    }

    const padL = 70;
    const padR = 40;
    const padT = 30;
    const padB = 95;
    const plotW = 920 - padL - padR;
    const plotH = 330 - padT - padB;

    const getX = (idx) => padL + (idx / (totalSteps - 1)) * plotW;
    const getY = (val) => padT + (1.0 - val) * plotH;

    let html = '';

    for (let i = 0; i < totalSteps; i++) {
        const xLeft = (i === 0) ? padL : (getX(i - 1) + getX(i)) / 2;
        const xRight = (i === totalSteps - 1) ? (padL + plotW) : (getX(i) + getX(i + 1)) / 2;
        const width = xRight - xLeft;
        const isSvr = evalHistory[i].hitter_is_svr;
        const fillColor = isSvr ? 'rgba(225, 29, 72, 0.05)' : 'rgba(2, 132, 199, 0.05)';
        html += `<rect x="${xLeft}" y="${padT}" width="${width}" height="${plotH}" fill="${fillColor}"/>`;
    }

    [0.0, 0.25, 0.50, 0.75, 1.0].forEach(level => {
        const y = getY(level);
        const isNeutral = (level === 0.50);
        html += `<line x1="${padL}" y1="${y}" x2="${padL + plotW}" y2="${y}" 
                       stroke="${isNeutral ? '#94a3b8' : 'rgba(255,255,255,0.08)'}" 
                       stroke-width="${isNeutral ? 1.5 : 1}" 
                       stroke-dasharray="${isNeutral ? '6,4' : '2,4'}"/>`;
        html += `<text x="${padL - 10}" y="${y + 4}" text-anchor="end" 
                       fill="${isNeutral ? '#f8fafc' : 'var(--text-muted)'}" 
                       font-size="${isNeutral ? '11' : '10'}" font-weight="${isNeutral ? 'bold' : 'normal'}">
                    ${(level * 100).toFixed(0)}%
                </text>`;
    });

    let dSvr = '';
    let dRet = '';
    evalHistory.forEach((h, idx) => {
        const x = getX(idx);
        const yS = getY(h.p_svr);
        const yR = getY(h.p_ret);
        dSvr += (idx === 0) ? `M ${x} ${yS}` : ` L ${x} ${yS}`;
        dRet += (idx === 0) ? `M ${x} ${yR}` : ` L ${x} ${yR}`;
    });

    html += `<path d="${dRet}" fill="none" stroke="#0284c7" stroke-width="3.0"/>`;
    html += `<path d="${dSvr}" fill="none" stroke="#e11d48" stroke-width="2.2" stroke-dasharray="4,3"/>`;

    evalHistory.forEach((h, idx) => {
        const x = getX(idx);
        const yS = getY(h.p_svr);
        const yR = getY(h.p_ret);
        const pSvrPct = (h.p_svr * 100).toFixed(0);
        const pRetPct = (h.p_ret * 100).toFixed(0);

        html += `<circle cx="${x}" cy="${yR}" r="4.5" fill="#0284c7" stroke="#ffffff" stroke-width="1.5"/>`;
        html += `<rect x="${x - 4}" y="${yS - 4}" width="8" height="8" fill="#e11d48" stroke="#ffffff" stroke-width="1.2"/>`;

        const svrHigher = (h.p_svr >= h.p_ret);
        const yTopLabel = svrHigher ? (yS - 14) : (yR - 14);
        const yBotLabel = svrHigher ? (yR + 18) : (yS + 18);
        const topVal = svrHigher ? `${pSvrPct}%` : `${pRetPct}%`;
        const botVal = svrHigher ? `${pRetPct}%` : `${pSvrPct}%`;
        const topColor = svrHigher ? '#e11d48' : '#0284c7';
        const botColor = svrHigher ? '#0284c7' : '#e11d48';

        html += `<text x="${x}" y="${yTopLabel}" text-anchor="middle" fill="${topColor}" font-size="10.5" font-weight="bold">${topVal}</text>`;
        html += `<text x="${x}" y="${yBotLabel}" text-anchor="middle" fill="${botColor}" font-size="10" font-weight="bold">${botVal}</text>`;

        const cleanLabel = h.label.replace(/\s*\([SR]\)\s*$/, '');
        html += `<text x="${x}" y="${padT + plotH + 18}" 
                       transform="rotate(26, ${x}, ${padT + plotH + 18})" 
                       text-anchor="start" fill="var(--text-secondary)" font-size="9" font-weight="500">
                    ${cleanLabel}
                </text>`;
    });

    svg.innerHTML = html;
}

function evaluateTacticalCategory(delta, hitterIsSvr) {
    const d = Math.abs(delta);
    if (d >= 0.20) return 'Tiro decisivo';
    if (d >= 0.10) return 'Impacto significativo';
    if (d >= 0.05) return 'Variación moderada';
    return 'Continuidad táctica';
}

function formatShotLabel(type, dir, depth, approach) {
    const tipos = {
        '4': 'Saque abierto', '5': 'Saque al cuerpo', '6': 'Saque a la T',
        'f': 'Derecha', 'b': 'Revés', 's': 'Slice de revés', 'r': 'Slice de derecha',
        'v': 'Volea', 'z': 'Media volea', 'o': 'Smash',
        't': 'Drop FH', 'u': 'Drop BH', 'g': 'Globo'
    };
    const dirs = { '1': 'paralela', '2': 'al centro', '3': 'cruzada' };
    const depths = { '7': 'corta', '8': 'media', '9': 'profunda' };

    let txt = tipos[type] || 'Tiro';
    if (dirs[dir]) txt += ` ${dirs[dir]}`;
    if (depths[depth]) txt += ` ${depths[depth]}`;
    if (approach) txt += ' con approach';
    return txt;
}

function setStatus(text, type = '') {
    const bar = document.getElementById('status-bar');
    if (!bar) return;
    bar.textContent = text;
    bar.className = 'status-bar' + (type ? ` status-${type}` : '');
}
