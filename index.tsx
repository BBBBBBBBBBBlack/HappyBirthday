import React, { useState, useEffect, useRef, useMemo, Suspense } from 'react';
import { createRoot } from 'react-dom/client';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Environment, OrbitControls, PerspectiveCamera } from '@react-three/drei';
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing';
import * as THREE from 'three';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

// --- Constants & Config ---
const PARTICLE_COUNT = 5000; // Increased slightly for better density
const SCATTER_RADIUS = 30;

// Specific Colors
const COLOR_WHITE = new THREE.Color('#FFFFFF'); // Metallic White
const COLOR_SILVER = new THREE.Color('#E0E0E0'); // Silver
const COLOR_RED = new THREE.Color('#D00000');   // Metallic Red
const COLOR_GOLD = new THREE.Color('#FFD700');  // Gold

// --- Helper Math Functions ---

// Generate a random point in a sphere
const getRandomSpherePoint = (radius: number) => {
  const u = Math.random();
  const v = Math.random();
  const theta = 2 * Math.PI * u;
  const phi = Math.acos(2 * v - 1);
  const r = Math.cbrt(Math.random()) * radius;
  const sinPhi = Math.sin(phi);
  return new THREE.Vector3(
    r * sinPhi * Math.cos(theta),
    r * sinPhi * Math.sin(theta),
    r * Math.cos(phi)
  );
};

// Generate attributes for the Cake Shape (Position & Color)
const getCakeAttribute = (index: number, total: number) => {
  const pos = new THREE.Vector3();
  let color = COLOR_WHITE;
  let scale = 1.0;

  // Definitions for tiers: [radius, height, bottomY]
  const tiers = [
    { r: 4.5, h: 2.0, y: -2.5 }, // Bottom
    { r: 3.2, h: 1.8, y: -0.5 }, // Middle
    { r: 2.0, h: 1.5, y: 1.3 },  // Top
  ];

  // Allocation ratios
  const candleCount = 250;
  const remaining = total - candleCount;
  
  // Distribute remaining particles across 3 tiers based on volume approx
  const tier1Count = Math.floor(remaining * 0.45);
  const tier2Count = Math.floor(remaining * 0.35);
  const tier3Count = remaining - tier1Count - tier2Count;

  let tierIndex = 0;
  let localIndex = 0;

  if (index < candleCount) {
    // --- CANDLES / FLAMES ---
    // Concentrated in the center top
    const r = Math.random() * 0.4;
    const angle = Math.random() * Math.PI * 2;
    pos.x = r * Math.cos(angle);
    pos.z = r * Math.sin(angle);
    // Height variation for a flame look
    pos.y = 2.8 + Math.random() * 1.5 + (1 - r) * 0.5; 
    
    color = COLOR_GOLD;
    scale = 0.8 + Math.random() * 0.5; // Flicker size
  } else {
    // --- CAKE BODY & DECORATIONS ---
    const cakeIndex = index - candleCount;
    
    if (cakeIndex < tier1Count) {
      tierIndex = 0;
      localIndex = cakeIndex;
    } else if (cakeIndex < tier1Count + tier2Count) {
      tierIndex = 1;
      localIndex = cakeIndex - tier1Count;
    } else {
      tierIndex = 2;
      localIndex = cakeIndex - tier1Count - tier2Count;
    }

    const { r: maxR, h, y } = tiers[tierIndex];

    // Determine if this particle is a "Decoration" (Red Accent)
    // We want ordered decorations. Let's say every 15th particle is a decoration on the surface.
    const isDecoration = localIndex % 15 === 0;

    if (isDecoration) {
      // --- DECORATION (Red Beads/Fruits) ---
      // Place strictly on the surface
      color = COLOR_RED;
      scale = 1.5; // Larger accent
      
      // Organize in spirals or rings
      const rings = 3; 
      const ringIndex = localIndex % rings;
      const heightStep = h / (rings + 1);
      
      const angle = localIndex * 0.5; // Spiral placement
      
      pos.x = maxR * Math.cos(angle);
      pos.z = maxR * Math.sin(angle);
      pos.y = y + (heightStep * (ringIndex + 1)) + (Math.random() * 0.1); // Slight jitter
      
    } else {
      // --- CAKE BODY (White/Silver) ---
      // Distribute throughout volume, but bias towards surface for solidity
      const angle = Math.random() * Math.PI * 2;
      // Sqrt for uniform distribution, but let's mix uniform and surface
      const radiusRand = Math.random();
      const radius = (radiusRand > 0.3 ? 0.8 + radiusRand * 0.2 : radiusRand) * maxR; // Bias to outer shell
      
      pos.x = radius * Math.cos(angle);
      pos.z = radius * Math.sin(angle);
      pos.y = y + Math.random() * h;
      
      // Add subtle noise to white color for texture
      color = Math.random() > 0.7 ? COLOR_SILVER : COLOR_WHITE;
      scale = 0.8 + Math.random() * 0.4;
    }
  }

  return { pos, color, scale };
};

// --- Components ---

const HandTracker = ({ onHandStateChange }: { onHandStateChange: (state: 'SCATTERED' | 'TREE_SHAPE') => void }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [loaded, setLoaded] = useState(false);
  const lastVideoTimeRef = useRef(-1);
  const landmarkerRef = useRef<HandLandmarker | null>(null);

  useEffect(() => {
    const initHandLandmarker = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
        );
        
        landmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1
        });
        setLoaded(true);
      } catch (e) {
        console.error("Failed to load MediaPipe:", e);
      }
    };

    initHandLandmarker();

    return () => {
      landmarkerRef.current?.close();
    };
  }, []);

  useEffect(() => {
    if (!loaded || !videoRef.current) return;

    const enableCam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.addEventListener('loadeddata', predict);
        }
      } catch (err) {
        console.error("Camera access denied:", err);
      }
    };

    let animationFrameId: number;

    const predict = async () => {
      if (!landmarkerRef.current || !videoRef.current) return;

      let startTimeMs = performance.now();
      
      if (videoRef.current.currentTime !== lastVideoTimeRef.current) {
        lastVideoTimeRef.current = videoRef.current.currentTime;
        const results = landmarkerRef.current.detectForVideo(videoRef.current, startTimeMs);

        if (results.landmarks && results.landmarks.length > 0) {
          const landmarks = results.landmarks[0];
          
          const wrist = landmarks[0];
          const tips = [8, 12, 16, 20];
          
          let avgDist = 0;
          tips.forEach(idx => {
            const tip = landmarks[idx];
            const d = Math.sqrt(
              Math.pow(tip.x - wrist.x, 2) + 
              Math.pow(tip.y - wrist.y, 2) + 
              Math.pow(tip.z - wrist.z, 2)
            );
            avgDist += d;
          });
          avgDist /= tips.length;

          if (avgDist < 0.25) {
            onHandStateChange('TREE_SHAPE');
          } else {
            onHandStateChange('SCATTERED');
          }
        }
      }
      animationFrameId = requestAnimationFrame(predict);
    };

    enableCam();

    return () => {
      if(videoRef.current && videoRef.current.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
      }
      cancelAnimationFrame(animationFrameId);
    };
  }, [loaded, onHandStateChange]);

  return (
    <div className="cam-preview">
      <video ref={videoRef} autoPlay playsInline muted style={{ transform: 'scaleX(-1)' }} />
    </div>
  );
};

const Particles = ({ targetState }: { targetState: 'SCATTERED' | 'TREE_SHAPE' }) => {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  
  // Data storage
  const particles = useMemo(() => {
    const tempParticles = [];

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      // 1. Dual Positions
      const scatterPos = getRandomSpherePoint(SCATTER_RADIUS);
      const { pos: cakePos, color, scale: cakeScale } = getCakeAttribute(i, PARTICLE_COUNT);

      // 2. Initial Setup
      const baseScale = 0.1 * cakeScale; // Adjust global particle size here
      
      tempParticles.push({
        currentPos: scatterPos.clone(),
        scatterPos,
        cakePos,
        rotation: new THREE.Euler(Math.random() * Math.PI, Math.random() * Math.PI, 0),
        rotationSpeed: (Math.random() - 0.5) * 0.02,
        baseScale,
        color,
        wobble: Math.random() * Math.PI * 2,
        speed: Math.random() * 0.05 + 0.02
      });
    }
    return tempParticles;
  }, []);

  // Initialize Colors once
  useEffect(() => {
    if (!meshRef.current) return;
    const dummy = new THREE.Object3D();
    
    particles.forEach((p, i) => {
      meshRef.current!.setColorAt(i, p.color);
      // Initialize Matrix
      dummy.position.copy(p.currentPos);
      dummy.scale.setScalar(p.baseScale);
      dummy.updateMatrix();
      meshRef.current!.setMatrixAt(i, dummy.matrix);
    });
    meshRef.current.instanceColor!.needsUpdate = true;
    meshRef.current.instanceMatrix!.needsUpdate = true;
  }, [particles]);

  // Animation Loop
  useFrame((state) => {
    if (!meshRef.current) return;

    const time = state.clock.getElapsedTime();
    const dummy = new THREE.Object3D();
    const isAggregating = targetState === 'TREE_SHAPE';

    particles.forEach((p, i) => {
      // Target selection
      const target = isAggregating ? p.cakePos : p.scatterPos;
      
      // Interpolate position (Lerp)
      const lerpFactor = isAggregating ? 0.05 : 0.03; // Snappy aggregation, floaty scatter
      p.currentPos.lerp(target, lerpFactor);

      // Add noise
      const hoverAmp = isAggregating ? 0.02 : 0.1; // Less movement when in cake shape
      const hoverY = Math.sin(time * 2 + p.wobble) * hoverAmp;
      const hoverX = Math.cos(time * 1.5 + p.wobble) * hoverAmp;

      dummy.position.copy(p.currentPos);
      dummy.position.x += hoverX;
      dummy.position.y += hoverY;

      // Rotate
      dummy.rotation.set(
        p.rotation.x + time * p.rotationSpeed,
        p.rotation.y + time * p.rotationSpeed,
        p.rotation.z
      );

      // Scale
      // Pulse effect for candles or all
      let currentScale = p.baseScale;
      if (isAggregating && p.color === COLOR_GOLD) {
         currentScale += Math.sin(time * 10 + p.wobble) * 0.05; // Flicker
      } else {
         currentScale += Math.sin(time * 3 + p.wobble) * 0.01;
      }
      
      dummy.scale.setScalar(currentScale);

      dummy.updateMatrix();
      meshRef.current!.setMatrixAt(i, dummy.matrix);
    });

    meshRef.current.instanceMatrix!.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, PARTICLE_COUNT]}>
      {/* High-res sphere for better gloss */}
      <icosahedronGeometry args={[1, 1]} /> 
      <meshStandardMaterial 
        toneMapped={false}
        roughness={0.15}
        metalness={1.0}
        emissiveIntensity={0.6}
      />
    </instancedMesh>
  );
};

const PostEffects = () => {
  return (
    <EffectComposer disableNormalPass>
      <Bloom 
        luminanceThreshold={0.6} 
        mipmapBlur 
        intensity={1.2} 
        radius={0.5}
      />
      <Vignette eskil={false} offset={0.1} darkness={1.1} />
    </EffectComposer>
  );
};

const Scene = ({ currentState }: { currentState: 'SCATTERED' | 'TREE_SHAPE' }) => {
  return (
    <>
      <PerspectiveCamera makeDefault position={[0, 4, 16]} fov={45} />
      <OrbitControls 
        enablePan={false} 
        maxPolarAngle={Math.PI / 1.5} 
        minDistance={8} 
        maxDistance={30} 
        autoRotate={currentState === 'TREE_SHAPE'}
        autoRotateSpeed={1.0}
      />
      
      {/* Lighting for Luxury Feel */}
      <ambientLight intensity={0.4} />
      {/* Main Key Light - Warm Gold */}
      <pointLight position={[10, 10, 10]} intensity={1.5} color="#FFD700" distance={50} decay={2} />
      {/* Fill Light - Cool White */}
      <pointLight position={[-10, 5, -10]} intensity={1.0} color="#FFFFFF" distance={50} decay={2} />
      {/* Rim Light - Reddish for drama */}
      <spotLight position={[0, 15, -5]} angle={0.6} penumbra={1} intensity={2.5} color="#FF4040" />

      <Environment preset="city" />

      <Particles targetState={currentState} />
      
      <PostEffects />
    </>
  );
};

const App = () => {
  const [appState, setAppState] = useState<'SCATTERED' | 'TREE_SHAPE'>('SCATTERED');
  const [showLoading, setShowLoading] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setShowLoading(false), 2000);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div style={{ width: '100vw', height: '100vh', background: '#050505' }}>
      
      {showLoading && (
        <div className="loading-overlay" style={{ opacity: showLoading ? 1 : 0 }}>
          <div style={{ color: '#d4af37' }}>
            <svg width="50" height="50" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" />
            </svg>
          </div>
          <div className="loading-text">Initializing Experience</div>
        </div>
      )}

      <Canvas gl={{ antialias: false, toneMapping: THREE.ReinhardToneMapping, toneMappingExposure: 1.5 }} dpr={[1, 2]}>
        <Suspense fallback={null}>
          <Scene currentState={appState} />
        </Suspense>
      </Canvas>

      <HandTracker onHandStateChange={setAppState} />
      
      <div className="instructions">
        <h2>
          <span>OPEN HAND</span> to Scatter &nbsp;|&nbsp; <span>FIST</span> to Celebrate
        </h2>
      </div>
    </div>
  );
};

const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(<App />);
}

export default App;