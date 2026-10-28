(function () {
  'use strict';
  var L = window.PowerLib;
  var $ = function (id) { return document.getElementById(id); };
  var state = { mode: 'n', metric: 'prop' };
  var FIELDS = ['base', 'mean', 'sd', 'effect', 'eff-kind', 'n', 'alpha', 'power', 'sides', 'ctrl', 'variants', 'bonf', 'rho', 'traffic'];

  var PRESETS = {
    hs_conv:  { metric: 'prop', base: 0.57, 'eff-kind': 'rel', effect: 20 },
    hs_visit: { metric: 'prop', base: 10.6, 'eff-kind': 'rel', effect: 10 },
    hs_spend: { metric: 'mean', mean: 0.653, sd: 11.6, 'eff-kind': 'rel', effect: 20 },
    checkout: { metric: 'prop', base: 3, 'eff-kind': 'rel', effect: 5 }
  };

  var fmt = function (x) { return Math.round(x).toLocaleString('en-US'); };
  var pct = function (x, d) { return (x * 100).toFixed(d == null ? 1 : d) + '%'; };
  var sig = function (x) { return x >= 100 ? fmt(x) : x >= 1 ? x.toFixed(2) : x.toPrecision(3); };

  function setSeg(id, v) {
    Array.prototype.forEach.call($(id).querySelectorAll('button'), function (b) { b.classList.toggle('on', b.dataset.v === v); });
  }

  function read() {
    var g = function (id) { return $(id).value; };
    var alpha0 = +g('alpha'), k = +g('variants'), ctrl = +g('ctrl') / 100;
    var rho = +g('rho');
    return {
      alpha: g('bonf') === '1' ? alpha0 / k : alpha0, alpha0: alpha0, k: k, power: +g('power'),
      twoSided: g('sides') === '2', ratio: (1 - ctrl) / ctrl, ctrl: ctrl,
      vr: rho * rho, rho: rho, nIn: Math.round(+g('n')), traffic: +g('traffic'),
      p: +g('base') / 100, mean: +g('mean'), sd: +g('sd'), effect: +g('effect'), rel: g('eff-kind') === 'rel'
    };
  }

  // absolute effect implied by the inputs
  function deltaOf(s) {
    if (state.metric === 'prop') return s.rel ? s.p * s.effect / 100 : s.effect / 100;  // abs = percentage points
    return s.rel ? s.mean * s.effect / 100 : s.effect;
  }

  function powerAt(s, delta, nC, vr) {
    var o = { alpha: s.alpha, twoSided: s.twoSided, ratio: s.ratio, varianceReduction: vr };
    return state.metric === 'prop' ? L.powerProportion(s.p, delta, nC, o) : L.powerMean(s.sd, delta, nC, o);
  }

  function compute() {
    var s = read(), o = { alpha: s.alpha, power: s.power, twoSided: s.twoSided, ratio: s.ratio, varianceReduction: s.vr };
    var out = { s: s }, prop = state.metric === 'prop';
    if (prop && !(s.p > 0 && s.p < 1)) throw new RangeError('Baseline rate must be between 0% and 100%.');
    if (!prop && !(s.sd > 0)) throw new RangeError('Standard deviation must be positive.');
    if (state.mode === 'n') {
      var delta = deltaOf(s);
      if (!(delta > 0)) throw new RangeError('Effect must be greater than zero.');
      var r = prop ? L.sampleSizeProportion(s.p, delta, o) : L.sampleSizeMean(s.sd, delta, o);
      var plain = prop ? L.sampleSizeProportion(s.p, delta, Object.assign({}, o, { varianceReduction: 0 })) : L.sampleSizeMean(s.sd, delta, Object.assign({}, o, { varianceReduction: 0 }));
      out.delta = delta; out.nC = r.nControl; out.nT = r.nTreatment; out.plainNC = plain.nControl;
    } else if (state.mode === 'mde') {
      if (!(s.nIn >= 2)) throw new RangeError('Need at least 2 users in the control group.');
      out.nC = s.nIn; out.nT = Math.ceil(s.nIn * s.ratio);
      out.delta = prop ? L.mdeProportion(s.p, s.nIn, o) : L.mdeMean(s.sd, s.nIn, o);
    } else {
      var d = deltaOf(s);
      if (!(d > 0)) throw new RangeError('Effect must be greater than zero.');
      if (!(s.nIn >= 2)) throw new RangeError('Need at least 2 users in the control group.');
      out.delta = d; out.nC = s.nIn; out.nT = Math.ceil(s.nIn * s.ratio);
      out.achieved = powerAt(s, d, s.nIn, s.vr);
    }
    if (prop && s.p + out.delta >= 1) throw new RangeError('Baseline plus effect must stay below 100%.');
    out.total = out.nC + s.k * out.nT;
    return out;
  }

  function render() {
    var err = $('r-err'); err.textContent = '';
    var out;
    try { out = compute(); } catch (e) { err.textContent = e.message; $('r-main').textContent = '–'; $('r-grid').innerHTML = ''; $('chart').innerHTML = ''; $('r-sub').textContent = ''; $('r-note').textContent = ''; return; }
    var s = out.s, prop = state.metric === 'prop';
    var base = prop ? s.p : s.mean, unit = prop ? ' pp' : '';
    var absTxt = prop ? (out.delta * 100).toFixed(2) + ' pp' : sig(out.delta);
    var relTxt = base ? pct(out.delta / base) : '';
    var title, main, sub;
    if (state.mode === 'n') {
      title = 'Required sample size'; main = fmt(out.nC) + ' per control group';
      sub = 'to detect ' + absTxt + (base ? ' (' + relTxt + ' relative)' : '') + ' with ' + pct(s.power, 0) + ' power at α = ' + s.alpha.toPrecision(2).replace(/0+$/, '');
    } else if (state.mode === 'mde') {
      title = 'Minimum detectable effect'; main = absTxt + (base ? ' (' + relTxt + ' relative)' : '');
      sub = 'with ' + fmt(out.nC) + ' control users and ' + pct(s.power, 0) + ' power';
    } else {
      title = 'Statistical power'; main = pct(out.achieved, 1);
      sub = 'chance of detecting ' + absTxt + ' with ' + fmt(out.nC) + ' control users';
    }
    $('r-title').textContent = title; $('r-main').textContent = main; $('r-sub').textContent = sub;

    var cells = [[fmt(out.nC), 'control'], [fmt(out.nT) + (s.k > 1 ? ' × ' + s.k : ''), 'per treatment'], [fmt(out.total), 'total users']];
    if (s.traffic > 0) cells[2] = [fmt(out.total) + ' · ' + Math.ceil(out.total / s.traffic) + ' d', 'total · days to run'];
    $('r-grid').innerHTML = cells.map(function (c) { return '<div><b>' + c[0] + '</b><span>' + c[1] + '</span></div>'; }).join('');

    var notes = [];
    if (s.k > 1 && $('bonf').value === '1') notes.push('α per comparison = ' + s.alpha.toPrecision(2) + ' (Bonferroni, ' + s.k + ' variants).');
    if (s.vr > 0 && out.plainNC) notes.push('CUPED saves ' + pct(1 - out.nC / out.plainNC, 0) + ' of users (' + fmt(out.plainNC) + ' without it).');
    $('r-note').textContent = notes.join(' ');
    drawChart(s, out);
  }

  function drawChart(s, out) {
    var W = 520, H = 300, m = { l: 46, r: 14, t: 14, b: 40 };
    var xmax = Math.max(out.nC * 2, 50), pts = 80, i, x, y;
    var X = function (n) { return m.l + n / xmax * (W - m.l - m.r); };
    var Y = function (p) { return H - m.b - p * (H - m.t - m.b); };
    var curve = function (vr) {
      var d = '';
      for (i = 1; i <= pts; i++) { x = xmax * i / pts; y = powerAt(s, out.delta, Math.max(2, x), vr); d += (i === 1 ? 'M' : 'L') + X(x).toFixed(1) + ' ' + Y(y).toFixed(1); }
      return d;
    };
    var svg = '';
    [0, 0.2, 0.4, 0.6, 0.8, 1].forEach(function (g) {
      svg += '<line x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(g) + '" y2="' + Y(g) + '" stroke="var(--line)"/>' +
             '<text x="' + (m.l - 6) + '" y="' + (Y(g) + 4) + '" text-anchor="end">' + Math.round(g * 100) + '%</text>';
    });
    [0, 0.5, 1].forEach(function (f) {
      svg += '<text x="' + X(xmax * f) + '" y="' + (H - m.b + 16) + '" text-anchor="' + (f === 0 ? 'start' : f === 1 ? 'end' : 'middle') + '">' + fmt(xmax * f) + '</text>';
    });
    svg += '<text x="' + (W / 2) + '" y="' + (H - 6) + '" text-anchor="middle">users in control group</text>';
    if (state.mode !== 'power') svg += '<line x1="' + m.l + '" x2="' + (W - m.r) + '" y1="' + Y(s.power) + '" y2="' + Y(s.power) + '" stroke="var(--muted)" stroke-dasharray="4 4"/>';
    if (s.vr > 0) svg += '<path d="' + curve(0) + '" fill="none" stroke="var(--muted)" stroke-width="2" stroke-dasharray="5 4"/>';
    svg += '<path d="' + curve(s.vr) + '" fill="none" stroke="var(--accent)" stroke-width="2.5"/>';
    var py = powerAt(s, out.delta, out.nC, s.vr);
    svg += '<circle cx="' + X(out.nC) + '" cy="' + Y(py) + '" r="5.5" fill="var(--accent)" stroke="var(--surface)" stroke-width="2"/>';
    svg += '<text x="' + (X(out.nC) + 10) + '" y="' + (Y(py) + 16) + '" style="fill:var(--ink);font-weight:600">' + fmt(out.nC) + ' → ' + pct(py, 0) + '</text>';
    if (s.vr > 0) svg += '<text x="' + (W - m.r) + '" y="' + (Y(0) - 8) + '" text-anchor="end">dashed: without CUPED</text>';
    $('chart').innerHTML = svg;
  }

  function sync() {
    var prop = state.metric === 'prop', mode = state.mode;
    $('f-base-prop').classList.toggle('hide', !prop);
    $('f-base-mean').classList.toggle('hide', prop);
    $('f-effect').classList.toggle('hide', mode === 'mde');
    $('f-n').classList.toggle('hide', mode === 'n');
    $('f-power').classList.toggle('hide', mode === 'power');
    $('f-traffic').classList.toggle('hide', mode === 'power');
    var ek = $('eff-kind'); ek.options[1].textContent = prop ? 'absolute (percentage points)' : 'absolute (metric units)';
    $('ctrl-v').textContent = $('ctrl').value + '%';
    $('rho-v').textContent = (+$('rho').value).toFixed(2) + (+$('rho').value > 0 ? ' → ' + Math.round(+$('rho').value * +$('rho').value * 100) + '% less variance' : '');
    setSeg('mode', mode); setSeg('metric', state.metric);
  }

  function save() {
    var q = new URLSearchParams({ mode: state.mode, metric: state.metric });
    FIELDS.forEach(function (f) { if ($(f).value !== '') q.set(f, $(f).value); });
    try { history.replaceState(null, '', '#' + q.toString()); } catch (e) { /* file:// or sandboxed */ }
  }

  function load() {
    var q = new URLSearchParams(location.hash.slice(1));
    if (q.get('mode')) state.mode = q.get('mode');
    if (q.get('metric')) state.metric = q.get('metric');
    FIELDS.forEach(function (f) { if (q.has(f)) $(f).value = q.get(f); });
  }

  function update() { sync(); render(); save(); }

  FIELDS.forEach(function (f) { $(f).addEventListener('input', update); });
  ['mode', 'metric'].forEach(function (id) {
    $(id).addEventListener('click', function (e) {
      if (!e.target.dataset.v) return;
      state[id === 'mode' ? 'mode' : 'metric'] = e.target.dataset.v; update();
    });
  });
  $('presets').addEventListener('click', function (e) {
    var p = PRESETS[e.target.dataset.preset]; if (!p) return;
    state.metric = p.metric;
    Object.keys(p).forEach(function (k) { if (k !== 'metric') $(k).value = p[k]; });
    update();
  });
  $('copy').addEventListener('click', function () {
    var b = $('copy'), done = function () { b.textContent = 'Copied ✓'; setTimeout(function () { b.textContent = 'Copy link to this setup'; }, 1500); };
    if (navigator.clipboard) navigator.clipboard.writeText(location.href).then(done, done); else done();
  });

  load(); update();
})();
