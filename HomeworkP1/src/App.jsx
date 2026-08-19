import React, { useState, useRef, useEffect, useCallback } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import './App.css';

// ─── Axis Arrow (custom colored arrow for Fixed Global Frame) ───────────────
function AxisArrow({ direction, color, label, length = 5 }) {
  const dir = new THREE.Vector3(...direction).normalize();
  const origin = new THREE.Vector3(0, 0, 0);
  return (
    <group>
      <arrowHelper args={[dir, origin, length, color, length * 0.12, length * 0.06]} />
      <Html position={direction.map(v => v * (length + 0.4))} center>
        <span style={{ color, fontWeight: 700, fontSize: '13px', userSelect: 'none' }}>{label}</span>
      </Html>
    </group>
  );
}

// ─── Fixed Reference Frame (stationary at origin) ────────────────────────────
function FixedFrame() {
  return (
    <group>
      <AxisArrow direction={[1, 0, 0]} color="#ff4444" label="X" length={5} />
      <AxisArrow direction={[0, 1, 0]} color="#44ff44" label="Y" length={5} />
      <AxisArrow direction={[0, 0, 1]} color="#4488ff" label="Z" length={5} />
    </group>
  );
}

// ─── Asymmetric Rigid Body (Only the actual 3D object) ───────────────────────
function AsymmetricBody() {
  return (
    <group>
      {/* Central core – white glass cube */}
      <mesh castShadow receiveShadow>
        <boxGeometry args={[0.55, 0.55, 0.55]} />
        <meshPhysicalMaterial
          color="#e0e8ff"
          transmission={0.7}
          transparent
          roughness={0.05}
          metalness={0.1}
          thickness={1.2}
        />
      </mesh>

      {/* Long arm along +X (red) */}
      <mesh position={[1.1, 0, 0]} castShadow receiveShadow>
        <boxGeometry args={[1.9, 0.25, 0.25]} />
        <meshPhysicalMaterial color="#ff6b6b" transmission={0.6} transparent roughness={0.1} thickness={0.8} />
      </mesh>

      {/* Tall arm along +Y (green) */}
      <mesh position={[0, 0.9, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.25, 1.55, 0.25]} />
        <meshPhysicalMaterial color="#51cf66" transmission={0.6} transparent roughness={0.1} thickness={0.8} />
      </mesh>

      {/* Short arm along +Z (blue) */}
      <mesh position={[0, 0, 0.6]} castShadow receiveShadow>
        <boxGeometry args={[0.25, 0.25, 0.95]} />
        <meshPhysicalMaterial color="#4dabf7" transmission={0.6} transparent roughness={0.1} thickness={0.8} />
      </mesh>

      {/* Small offset nub – makes it clearly asymmetric */}
      <mesh position={[0.55, 0.55, -0.35]} castShadow receiveShadow>
        <sphereGeometry args={[0.18, 12, 12]} />
        <meshPhysicalMaterial color="#ffd43b" transmission={0.5} transparent roughness={0.15} thickness={0.5} />
      </mesh>
    </group>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [rotation, setRotation] = useState({ x: 0, y: 0, z: 0 });
  const [position, setPosition] = useState({ x: 2.5, y: 1.5, z: 0 });
  const [mode, setMode] = useState('rotate'); // 'rotate' | 'translate'
  const [activeKeyHint, setActiveKeyHint] = useState(null); // feedback for pressed axis key
  const orbitRef = useRef();

  // Track currently pressed keys for multi-key combos (e.g. X + Up)
  const pressedKeysRef = useRef(new Set());
  const lastAxisRef = useRef('x');

  // Derive live rotation matrix from current euler angles
  const euler = new THREE.Euler(rotation.x, rotation.y, rotation.z, 'XYZ');
  const rotMatrix = new THREE.Matrix4().makeRotationFromEuler(euler);
  const el = rotMatrix.elements; // column-major

  // Row-major extraction for display: row i, col j → el[j*4 + i]
  const matRows = [
    [el[0], el[4], el[8]],
    [el[1], el[5], el[9]],
    [el[2], el[6], el[10]],
  ];

  // Handle direct matrix cell edits
  const handleMatrixEdit = useCallback((row, col, value) => {
    const num = parseFloat(value);
    if (isNaN(num)) return;
    const idx = col * 4 + row;
    const newEl = [...el];
    newEl[idx] = num;
    // Build a Matrix4, extract euler
    const m = new THREE.Matrix4();
    m.elements = newEl;
    const newEuler = new THREE.Euler().setFromRotationMatrix(m, 'XYZ');
    if (isFinite(newEuler.x) && isFinite(newEuler.y) && isFinite(newEuler.z)) {
      setRotation({ x: newEuler.x, y: newEuler.y, z: newEuler.z });
    }
  }, [el]);

  // Handle keyboard controls: M toggles mode; X/Y/Z + Up/Down moves or rotates
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Ignore key events when user is typing in form inputs
      if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;

      const key = e.key.toLowerCase();
      pressedKeysRef.current.add(key);

      // 'M' toggles Move/Rotate mode
      if (key === 'm') {
        setMode(m => (m === 'translate' ? 'rotate' : 'translate'));
        return;
      }

      // Track axis key
      if (['x', 'y', 'z'].includes(key)) {
        lastAxisRef.current = key;
        setActiveKeyHint(key.toUpperCase());
      }

      // Arrow Up / Down triggers transformation on the active axis
      if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
        e.preventDefault();
        const delta = e.key === 'ArrowUp' ? 1 : -1;

        // Check if X, Y, or Z key is currently held, otherwise use the last selected axis
        let targetAxis = null;
        if (pressedKeysRef.current.has('x')) targetAxis = 'x';
        else if (pressedKeysRef.current.has('y')) targetAxis = 'y';
        else if (pressedKeysRef.current.has('z')) targetAxis = 'z';
        else targetAxis = lastAxisRef.current || 'x';

        setActiveKeyHint(`${targetAxis.toUpperCase()} ${delta > 0 ? '▲' : '▼'}`);

        if (mode === 'translate') {
          // Move object along target axis by 0.1
          const step = 0.1;
          setPosition(p => ({
            ...p,
            [targetAxis]: parseFloat((p[targetAxis] + delta * step).toFixed(2))
          }));
        } else {
          // Rotate object around target axis by ~2.86° (0.05 rad)
          const step = 0.05;
          setRotation(r => {
            let nextVal = r[targetAxis] + delta * step;
            // Wrap between -PI and PI
            if (nextVal > Math.PI) nextVal -= 2 * Math.PI;
            if (nextVal < -Math.PI) nextVal += 2 * Math.PI;
            return {
              ...r,
              [targetAxis]: parseFloat(nextVal.toFixed(3))
            };
          });
        }
      }
    };

    const handleKeyUp = (e) => {
      const key = e.key.toLowerCase();
      pressedKeysRef.current.delete(key);
      if (['x', 'y', 'z'].includes(key)) {
        // Clear hint after short delay
        setTimeout(() => setActiveKeyHint(null), 800);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [mode]);

  const resetOrientation = () => setRotation({ x: 0, y: 0, z: 0 });
  const resetPosition = () => setPosition({ x: 2.5, y: 1.5, z: 0 });

  const axisColors = { X: '#ff8787', Y: '#69db7c', Z: '#74c0fc' };

  return (
    <div className="app-root">
      {/* ── Side Panel ─────────────────────────────────────────────── */}
      <aside className="panel">
        <h2 className="panel-title">Rigid Body Visualizer</h2>
        <p className="panel-hint">
          {mode === 'rotate'
            ? 'Rotate Mode: Adjust sliders, edit the matrix, or use X/Y/Z + Up/Down arrows.'
            : 'Move Mode: Adjust position sliders or use X/Y/Z + Up/Down arrows to translate in 3D.'}
        </p>

        {/* In Rotate Mode: Show only Rotation Sliders, Rotation Matrix, and Reset Orientation */}
        {mode === 'rotate' ? (
          <>
            {/* Angle Sliders */}
            <section className="section">
              <h3 className="section-title">Euler Angles (XYZ)</h3>
              {['x', 'y', 'z'].map((axis) => (
                <div key={axis} className="slider-row">
                  <span className="slider-label" style={{ color: axisColors[axis.toUpperCase()] }}>
                    {axis.toUpperCase()}
                  </span>
                  <input
                    type="range"
                    className="slider"
                    min={-Math.PI}
                    max={Math.PI}
                    step={0.01}
                    value={rotation[axis]}
                    onChange={(e) => setRotation(r => ({ ...r, [axis]: parseFloat(e.target.value) }))}
                  />
                  <span className="slider-value">{(rotation[axis] * (180 / Math.PI)).toFixed(1)}°</span>
                </div>
              ))}
            </section>

            {/* Editable Rotation Matrix */}
            <section className="section">
              <h3 className="section-title">Rotation Matrix R</h3>
              <div className="matrix-grid">
                {matRows.map((row, ri) =>
                  row.map((val, ci) => (
                    <input
                      key={`${ri}-${ci}`}
                      type="number"
                      className="matrix-cell"
                      step="0.01"
                      value={parseFloat(val.toFixed(4))}
                      onChange={(e) => handleMatrixEdit(ri, ci, e.target.value)}
                    />
                  ))
                )}
              </div>
              <p className="matrix-note">Columns: x_b, y_b, z_b expressed in fixed frame</p>
            </section>

            {/* Keyboard Guide */}
            <section className="section keyboard-guide-section">
              <h3 className="section-title">Keyboard Controls</h3>
              <div className="guide-row">
                <span className="guide-key"><kbd>M</kbd></span>
                <span className="guide-desc">Toggle Move / Rotate</span>
              </div>
              <div className="guide-row">
                <span className="guide-key"><kbd>X</kbd>/<kbd>Y</kbd>/<kbd>Z</kbd> + <kbd>↑</kbd>/<kbd>↓</kbd></span>
                <span className="guide-desc">Rotate along axis</span>
              </div>
            </section>

            {/* Reset Button */}
            <button className="reset-btn" onClick={resetOrientation}>
              ↺ Reset Orientation
            </button>
          </>
        ) : (
          <>
            {/* In Move Mode: Show Position (Translation) Controls */}
            <section className="section">
              <h3 className="section-title">Position (Translation)</h3>
              {['x', 'y', 'z'].map((axis) => (
                <div key={axis} className="slider-row">
                  <span className="slider-label" style={{ color: axisColors[axis.toUpperCase()] }}>
                    {axis.toUpperCase()}
                  </span>
                  <input
                    type="range"
                    className="slider"
                    min={axis === 'y' ? -2 : -8}
                    max={8}
                    step={0.05}
                    value={position[axis]}
                    onChange={(e) => setPosition(p => ({ ...p, [axis]: parseFloat(e.target.value) }))}
                  />
                  <span className="slider-value">{position[axis].toFixed(2)}</span>
                </div>
              ))}
            </section>

            {/* Keyboard Guide */}
            <section className="section keyboard-guide-section">
              <h3 className="section-title">Keyboard Controls</h3>
              <div className="guide-row">
                <span className="guide-key"><kbd>M</kbd></span>
                <span className="guide-desc">Toggle Move / Rotate</span>
              </div>
              <div className="guide-row">
                <span className="guide-key"><kbd>X</kbd>/<kbd>Y</kbd>/<kbd>Z</kbd> + <kbd>↑</kbd>/<kbd>↓</kbd></span>
                <span className="guide-desc">Move along axis (±0.1)</span>
              </div>
            </section>

            {/* Reset Position Button */}
            <button className="reset-btn reset-pos-btn" onClick={resetPosition}>
              ↺ Reset Position
            </button>
          </>
        )}
      </aside>

      {/* ── 3D Canvas Area with Overlays ────────────────────────────── */}
      <div className="canvas-wrapper">
        {/* ── Top-Right Move Mode Button Control ──────────────────── */}
        <div className={`top-right-move-container ${mode === 'translate' ? 'is-active' : ''}`}>
          <div className="top-right-bar">
            <button
              className={`top-mode-btn ${mode === 'translate' ? 'active-move' : ''}`}
              id="top-right-move-btn"
              onClick={() => setMode(m => (m === 'translate' ? 'rotate' : 'translate'))}
              title="Toggle Move Mode (Shortcut: M)"
            >
              <div className="btn-inner">
                <span className="mode-btn-icon">✥</span>
                <div className="mode-btn-text-block">
                  <span className="mode-btn-title">Move Mode</span>
                  <span className="mode-btn-subtitle">
                    {mode === 'translate' ? 'ACTIVE • X/Y/Z + ↑/↓ to move' : 'Click or press M to activate'}
                  </span>
                </div>
                <div className={`status-pill ${mode === 'translate' ? 'pill-on' : 'pill-off'}`}>
                  <span className="pulse-dot" />
                  <span>{mode === 'translate' ? 'ON' : 'OFF'}</span>
                </div>
              </div>
            </button>
            <div className="top-bar-subrow">
              <span className="shortcut-hint">
                <kbd>M</kbd> toggle mode • <kbd>X</kbd>/<kbd>Y</kbd>/<kbd>Z</kbd> + <kbd>↑</kbd>/<kbd>↓</kbd> to {mode === 'translate' ? 'move' : 'rotate'}
              </span>
            </div>
          </div>
        </div>

        {/* ── Active Move Mode Floating HUD Banner ────────────────── */}
        {/* {mode === 'translate' && (
          <div className="canvas-hud-banner">
            <div className="hud-badge">
              <span className="hud-pulse" />
              <span className="hud-icon">✥</span>
              <span className="hud-title">MOVE MODE ACTIVE</span>
            </div>
            <div className="hud-description">
              Press <strong><kbd>X</kbd>, <kbd>Y</kbd>, or <kbd>Z</kbd> + <kbd>↑</kbd>/<kbd>↓</kbd></strong> to move the object. Press <kbd>M</kbd> to exit.
            </div>
          </div>
        )} */}

        {/* ── Active Key Visual Indicator (feedback for keyboard input) ── */}
        {activeKeyHint && (
          <div className="key-feedback-chip">
            <span>Active Axis: <strong>{activeKeyHint}</strong></span>
          </div>
        )}

        <Canvas
          shadows
          camera={{ position: [7, 5, 9], fov: 45 }}
          className="canvas"
        >
          <color attach="background" args={['#0e0e14']} />
          <fog attach="fog" args={['#0e0e14', 15, 45]} />

          {/* Lighting */}
          <ambientLight intensity={0.8} />
          <directionalLight
            position={[12, 18, 10]}
            intensity={3}
            castShadow
            shadow-mapSize-width={2048}
            shadow-mapSize-height={2048}
            shadow-camera-left={-12}
            shadow-camera-right={12}
            shadow-camera-top={12}
            shadow-camera-bottom={-12}
          />
          <pointLight position={[-8, 6, -8]} intensity={1.2} color="#8ab4f8" />
          <pointLight position={[6, -2, 6]} intensity={0.6} color="#f8a0a0" />

          {/* Ground grid */}
          <gridHelper args={[40, 40, '#2a2a3a', '#1a1a28']} position={[0, -0.01, 0]} />

          {/* Shadow receiver */}
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.015, 0]} receiveShadow>
            <planeGeometry args={[100, 100]} />
            <shadowMaterial transparent opacity={0.4} />
          </mesh>

          {/* Fixed global reference frame at origin */}
          <FixedFrame />

          {/* ── Only the actual 3D object itself ─────────────────────── */}
          <group
            position={[position.x, position.y, position.z]}
            rotation={[rotation.x, rotation.y, rotation.z]}
          >
            <AsymmetricBody />
          </group>

          <OrbitControls ref={orbitRef} makeDefault enableDamping dampingFactor={0.08} />
        </Canvas>
      </div>
    </div>
  );
}
