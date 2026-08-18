import React, { useState, useRef, useEffect, useCallback } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, TransformControls, Html } from '@react-three/drei';
import './App.css';

// ─── Axis Arrow (custom colored arrow) ───────────────────────────────────────
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

// ─── Fixed Reference Frame ────────────────────────────────────────────────────
function FixedFrame() {
  return (
    <group>
      <AxisArrow direction={[1, 0, 0]} color="#ff4444" label="X" length={5} />
      <AxisArrow direction={[0, 1, 0]} color="#44ff44" label="Y" length={5} />
      <AxisArrow direction={[0, 0, 1]} color="#4488ff" label="Z" length={5} />
    </group>
  );
}

// ─── Body Frame Axes (moves with the rigid body) ──────────────────────────────
function BodyFrame() {
  return (
    <group>
      <AxisArrow direction={[1, 0, 0]} color="#ff8787" label="x_b" length={2.2} />
      <AxisArrow direction={[0, 1, 0]} color="#69db7c" label="y_b" length={2.2} />
      <AxisArrow direction={[0, 0, 1]} color="#74c0fc" label="z_b" length={2.2} />
    </group>
  );
}

// ─── Asymmetric Rigid Body ────────────────────────────────────────────────────
function AsymmetricBody() {
  return (
    <group>
      {/* Central core – white glass cube */}
      <mesh castShadow receiveShadow>
        <boxGeometry args={[0.55, 0.55, 0.55]} />
        <meshPhysicalMaterial
          color="#e0e8ff" transmission={0.7} transparent
          roughness={0.05} metalness={0.1} thickness={1.2}
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

// ─── Rigid Body Group (controlled by rotation state) ─────────────────────────
function RigidBody({ rotation, setRotation, orbitRef }) {
  const groupRef = useRef();

  // Sync group rotation when sliders / matrix change
  useEffect(() => {
    if (groupRef.current) {
      groupRef.current.rotation.set(rotation.x, rotation.y, rotation.z, 'XYZ');
    }
  }, [rotation.x, rotation.y, rotation.z]);

  return (
    <TransformControls
      mode="rotate"
      position={[2.5, 1.5, 0]}
      onObjectChange={(e) => {
        if (e?.target?.object) {
          const { x, y, z } = e.target.object.rotation;
          setRotation({ x, y, z });
        }
      }}
      // Prevent gizmo drag from also panning the camera
      onMouseDown={() => { if (orbitRef.current) orbitRef.current.enabled = false; }}
      onMouseUp={() => { if (orbitRef.current) orbitRef.current.enabled = true; }}
    >
      <group ref={groupRef}>
        <AsymmetricBody />
        <BodyFrame />
      </group>
    </TransformControls>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [rotation, setRotation] = useState({ x: 0, y: 0, z: 0 });
  const orbitRef = useRef();

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

  const reset = () => setRotation({ x: 0, y: 0, z: 0 });

  const axisColors = { X: '#ff8787', Y: '#69db7c', Z: '#74c0fc' };

  return (
    <div className="app-root">
      {/* ── Side Panel ─────────────────────────────────────────────── */}
      <aside className="panel">
        <h2 className="panel-title">Rigid Body Visualizer</h2>
        <p className="panel-hint">Drag the gizmo rings in the scene, adjust the sliders, or edit the matrix directly.</p>

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
              {/* <span className="slider-rad">{rotation[axis].toFixed(3)}</span> */}
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

        {/* Reset Button */}
        <button className="reset-btn" onClick={reset}>
          ↺ Reset Orientation
        </button>
      </aside>

      {/* ── 3D Canvas ──────────────────────────────────────────────── */}
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

        {/* Fixed global frame at origin */}
        <FixedFrame />

        {/* Rotatable rigid body + body frame */}
        <RigidBody
          rotation={rotation}
          setRotation={setRotation}
          orbitRef={orbitRef}
        />

        <OrbitControls ref={orbitRef} makeDefault enableDamping dampingFactor={0.08} />
      </Canvas>
    </div>
  );
}
