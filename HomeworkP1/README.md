# 3D Rigid Body Rotation Visualizer

## Overview
This project is an interactive, real-time 3D web application built with React and Three.js (via React Three Fiber). It serves as a visual playground for understanding rigid body kinematics, specifically focusing on the relationship between 3D orientations, Euler angles (XYZ), and 3×3 rotation matrices.

## Purpose
Understanding 3D rotations can be notoriously difficult due to the mathematical abstraction of rotation matrices and the non-intuitive nature of Euler angles (e.g., gimbal lock, axis order). The purpose of this tool is to bridge the gap between the math and visual intuition. By allowing users to interact with a 3D object and immediately see the underlying mathematical representations update in real-time, it provides a hands-on learning environment for students, engineers, and developers working with 3D graphics, robotics, or physics engines.

## Key Features
*   **Real-Time Bi-directional Synchronization:** The application state is fully synchronized. You can rotate the object by dragging the 3D gizmo, adjusting the UI sliders, or directly typing values into the 3×3 rotation matrix. Updating one instantly updates the others.
*   **Distinct Coordinate Frames:** The scene clearly delineates the **Fixed Global Frame** (stationary at the origin) and the **Local Body Frame** (moves and rotates with the object).
*   **Interactive 3D Canvas:** Built-in OrbitControls allow you to pan and zoom around the scene, while TransformControls allow direct manipulation of the object's rotation.
*   **Modern UI:** A sleek, non-obtrusive "glassmorphism" side panel houses the controls, maximizing the viewable area of the 3D canvas.

## Design & Implementation Decisions (The "Why")

### 1. The Asymmetric 3D Object
Instead of using a simple, standard primitive like a cube or a sphere, the central object is a custom, highly asymmetric composite shape (resembling an intricate tool or spacecraft, with distinct arms extending along the X, Y, and Z axes). 
*   **Why?** Symmetric objects suffer from visual aliasing—if you rotate a perfect cube by 90 degrees, it looks identical to its unrotated state. By making the object completely asymmetric (e.g., a long red arm on X, a tall green arm on Y, a short blue arm on Z, and an offset yellow nub), every possible orientation is visually unique. This eliminates ambiguity and makes it instantly clear how the body is oriented in space.

### 2. Standardized Color Coding
The X, Y, and Z axes are strictly color-coded to Red, Green, and Blue, respectively.
*   **Why?** This adheres to the industry standard in 3D computer graphics (RGB = XYZ). We further distinguished the frames by using highly saturated primary colors for the **Fixed Frame**, and softer, pastel variations for the **Body Frame**. This prevents visual clutter while maintaining the intuitive RGB mapping.

### 3. State-Driven Matrix Orthogonalization
The 3×3 rotation matrix in the UI is fully editable.
*   **Why?** Allowing direct matrix edits helps users understand how specific matrix elements affect orientation. 
*   **The Catch:** A valid rotation matrix must be orthogonal (its determinant must be 1). If a user types an arbitrary number into the matrix, it could introduce shearing or scaling. 
*   **The Solution:** By extracting Euler angles from the edited matrix (`setFromRotationMatrix`) and pushing those angles back to the central React state, the math inherently strips out invalid scaling/shearing. The system "self-corrects" user input into the nearest valid rotation matrix, ensuring the 3D body remains rigid.

### 4. React Three Fiber & Drei over Vanilla Three.js
The 3D scene is constructed declaratively using `@react-three/fiber` and `@react-three/drei`.
*   **Why?** In a standard vanilla Three.js setup, synchronizing HTML UI state with the 3D render loop often requires messy imperative code and manual update functions. React Three Fiber allows the 3D objects to react to the exact same React state hooks (`useState`) that power the HTML sliders and inputs. This guarantees flawless synchronization and drastically simplifies the architecture.

### 5. High-Fidelity Environment
The scene includes a dark slate background, depth fog, a shadow-catching floor, and glass-like Physical Materials (`MeshPhysicalMaterial`) with transmission and thickness.
*   **Why?** While a flat, unlit scene would function mathematically, adding high-quality lighting, shadows, and physical materials provides essential depth cues. The shadows cast on the floor plane anchor the object in space, helping the user accurately perceive the 3D rotation on a 2D screen.

## Setup & Installation

This project is built using React and Vite.

1.  **Install Dependencies:**
    Make sure you have Node.js installed, then run:
    ```bash
    npm install
    ```
2.  **Run the Development Server:**
    ```bash
    npm run dev
    ```
3.  **View the App:**
    Open your browser and navigate to the local URL provided by Vite (usually `http://localhost:5173`).

## Dependencies
*   `react` & `react-dom`
*   `three`: The core 3D rendering engine.
*   `@react-three/fiber`: React renderer for Three.js.
*   `@react-three/drei`: Useful helpers and abstractions for React Three Fiber.
