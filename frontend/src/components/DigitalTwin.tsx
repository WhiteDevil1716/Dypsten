import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import './DigitalTwin.css';

const DigitalTwin: React.FC = () => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [loading, setLoading] = useState(true);
    const [riskLevel, setRiskLevel] = useState('LOW');

    useEffect(() => {
        if (!containerRef.current) return;

        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0f1a);
        scene.fog = new THREE.Fog(0x0a0f1a, 10, 100);

        // Camera
        const camera = new THREE.PerspectiveCamera(
            60,
            containerRef.current.clientWidth / containerRef.current.clientHeight,
            0.1,
            1000
        );
        camera.position.set(50, 40, 50);

        // Renderer
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        containerRef.current.appendChild(renderer.domElement);

        // Controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.minDistance = 20;
        controls.maxDistance = 150;
        controls.maxPolarAngle = Math.PI / 2.1;

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
        scene.add(ambientLight);

        const

            directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(50, 80, 40);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        directionalLight.shadow.camera.far = 200;
        scene.add(directionalLight);

        // Add hemisphere light for natural lighting
        const hemisphereLight = new THREE.HemisphereLight(0x87ceeb, 0x543a2c, 0.6);
        scene.add(hemisphereLight);

        // Create terrain mesh
        const createTerrain = async () => {
            try {
                const response = await fetch('http://localhost:8000/api/terrain/mesh?size=64');
                const data = await response.json();

                // Create geometry from API data
                const geometry = new THREE.BufferGeometry();

                // Convert vertices array
                const vertices = new Float32Array(data.vertices.flat());
                geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

                // Convert faces (indices)
                const indices = new Uint32Array(data.faces.flat());
                geometry.setIndex(new THREE.BufferAttribute(indices, 1));

                // Convert normals
                const normals = new Float32Array(data.normals.flat());
                geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));

                // Convert colors
                const colors = new Float32Array(data.colors.flat());
                geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

                // Material with vertex colors
                const material = new THREE.MeshStandardMaterial({
                    vertexColors: true,
                    roughness: 0.8,
                    metalness: 0.2,
                    flatShading: false,
                });

                const terrain = new THREE.Mesh(geometry, material);
                terrain.castShadow = true;
                terrain.receiveShadow = true;
                terrain.rotation.x = -Math.PI / 2;
                terrain.position.y = 0;
                scene.add(terrain);

                // Add grid helper
                const gridHelper = new THREE.GridHelper(100, 20, 0x3b82f6, 0x1e40af);
                gridHelper.material.opacity = 0.2;
                gridHelper.material.transparent = true;
                scene.add(gridHelper);

                // Add risk overlay (heat map planes)
                createRiskOverlay(data.risk_map);

                setLoading(false);
            } catch (error) {
                console.error('Failed to load terrain:', error);
                // Create fallback procedural terrain
                createFallbackTerrain();
                setLoading(false);
            }
        };

        const createFallbackTerrain = () => {
            const geometry = new THREE.PlaneGeometry(80, 80, 50, 50);
            const vertices = geometry.attributes.position.array;

            // Create hills
            for (let i = 0; i < vertices.length; i += 3) {
                const x = vertices[i];
                const y = vertices[i + 1];
                vertices[i + 2] = Math.sin(x * 0.1) * Math.cos(y * 0.1) * 5 + Math.random() * 2;
            }

            geometry.computeVertexNormals();

            const material = new THREE.MeshStandardMaterial({
                color: 0x22c55e,
                roughness: 0.8,
                metalness: 0.2,
            });

            const terrain = new THREE.Mesh(geometry, material);
            terrain.rotation.x = -Math.PI / 2;
            terrain.castShadow = true;
            terrain.receiveShadow = true;
            scene.add(terrain);
        };

        const createRiskOverlay = (riskMap: number[][]) => {
            // Create heat map visualization
            const size = riskMap.length;
            const canvas = document.createElement('canvas');
            canvas.width = size;
            canvas.height = size;
            const ctx = canvas.getContext('2d')!;

            // Draw risk map
            for (let y = 0; y < size; y++) {
                for (let x = 0; x < size; x++) {
                    const risk = riskMap[y][x];
                    const color = getRiskColor(risk);
                    ctx.fillStyle = color;
                    ctx.fillRect(x, y, 1, 1);
                }
            }

            // Create texture from canvas
            const texture = new THREE.CanvasTexture(canvas);
            texture.needsUpdate = true;

            // Create overlay plane
            const overlayGeometry = new THREE.PlaneGeometry(80, 80);
            const overlayMaterial = new THREE.MeshBasicMaterial({
                map: texture,
                transparent: true,
                opacity: 0.6,
                side: THREE.DoubleSide,
            });

            const overlay = new THREE.Mesh(overlayGeometry, overlayMaterial);
            overlay.rotation.x = -Math.PI / 2;
            overlay.position.y = 0.1; // Slightly above terrain
            scene.add(overlay);
        };

        const getRiskColor = (risk: number): string => {
            if (risk < 25) return 'rgba(34, 197, 94, 0.5)'; // Green
            if (risk < 60) return 'rgba(245, 158, 11, 0.6)'; // Yellow
            if (risk < 85) return 'rgba(239, 68, 68, 0.7)'; // Red
            return 'rgba(220, 38, 38, 0.8)'; // Dark Red
        };

        // Add coordinate axes
        const axesHelper = new THREE.AxesHelper(30);
        scene.add(axesHelper);

        // Animation loop
        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };

        // Handle resize
        const handleResize = () => {
            if (!containerRef.current) return;

            camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
        };

        window.addEventListener('resize', handleResize);

        // Start
        createTerrain();
        animate();

        // Cleanup
        return () => {
            window.removeEventListener('resize', handleResize);
            controls.dispose();
            renderer.dispose();
            if (containerRef.current && renderer.domElement.parentNode === containerRef.current) {
                containerRef.current.removeChild(renderer.domElement);
            }
        };
    }, []);

    return (
        <div className="digital-twin-container">
            {loading && (
                <div className="loading-overlay">
                    <div className="spinner"></div>
                    <p>Loading terrain data...</p>
                </div>
            )}

            <div ref={containerRef} className="canvas-container" />

            <div className="twin-controls">
                <div className="view-controls">
                    <button className="view-btn" title="Top View">⬆️</button>
                    <button className="view-btn" title="Side View">↔️</button>
                    <button className="view-btn" title="3D View">🔄</button>
                </div>

                <div className="legend">
                    <h4>Risk Levels</h4>
                    <div className="legend-item">
                        <span className="legend-color" style={{ background: '#22c55e' }}></span>
                        <span>Low</span>
                    </div>
                    <div className="legend-item">
                        <span className="legend-color" style={{ background: '#f59e0b' }}></span>
                        <span>Medium</span>
                    </div>
                    <div className="legend-item">
                        <span className="legend-color" style={{ background: '#ef4444' }}></span>
                        <span>High</span>
                    </div>
                    <div className="legend-item">
                        <span className="legend-color" style={{ background: '#dc2626' }}></span>
                        <span>Critical</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default DigitalTwin;
