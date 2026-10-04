window.pumpViewer = {
    viewers: new Map(),

    initialize: async function (canvasId, modelUrl) {
        const canvas = document.getElementById(canvasId);

        if (!canvas) {
            console.error("3D canvas not found:", canvasId);
            return;
        }

        const engine = new BABYLON.Engine(canvas, true, {
            preserveDrawingBuffer: true,
            stencil: true
        });

        const scene = new BABYLON.Scene(engine);
        scene.clearColor = new BABYLON.Color4(0.03, 0.08, 0.12, 1);

        const camera = new BABYLON.ArcRotateCamera(
            "pumpCamera",
            Math.PI / 2,
            Math.PI / 3,
            10,
            BABYLON.Vector3.Zero(),
            scene
        );

        camera.attachControl(canvas, true);
        camera.wheelDeltaPercentage = 0.01;
        camera.panningSensibility = 100;

        scene.imageProcessingConfiguration.exposure = 1.2;
        scene.imageProcessingConfiguration.contrast = 1.05;

        const upperLight = new BABYLON.HemisphericLight(
            "upperLight",
            new BABYLON.Vector3(0, 1, 0),
            scene
        );

        upperLight.intensity = 1.1;
        upperLight.groundColor = new BABYLON.Color3(0.35, 0.4, 0.45);

        const lowerLight = new BABYLON.HemisphericLight(
            "lowerLight",
            new BABYLON.Vector3(0, -1, 0),
            scene
        );

        lowerLight.intensity = 0.7;
        lowerLight.groundColor = new BABYLON.Color3(0.3, 0.35, 0.4);

        const cameraLight = new BABYLON.PointLight(
            "cameraLight",
            BABYLON.Vector3.Zero(),
            scene
        );

        cameraLight.parent = camera;
        cameraLight.position = BABYLON.Vector3.Zero();
        cameraLight.intensity = 0.9;
        cameraLight.range = 100000;

        const result = await BABYLON.SceneLoader.ImportMeshAsync(
            null,
            "",
            modelUrl,
            scene
        );

        const modelMeshes = result.meshes.filter(
            mesh => mesh.getTotalVertices() > 0
        );

        if (modelMeshes.length === 0) {
            console.error("No visible meshes were found in the GLB file.");
            return;
        }

        modelMeshes.forEach(mesh => mesh.computeWorldMatrix(true));

        const bounds = scene.getWorldExtends(
            mesh => mesh.getTotalVertices() > 0
        );

        const minimum = bounds.min;
        const maximum = bounds.max;
        const center = minimum.add(maximum).scale(0.5);
        const size = maximum.subtract(minimum);
        const largestSize = Math.max(size.x, size.y, size.z);

        camera.target = center;
        camera.radius = largestSize * 1.3;
        camera.lowerRadiusLimit = largestSize * 0.25;
        camera.upperRadiusLimit = largestSize * 4;

        const glow = new BABYLON.GlowLayer("faultGlow", scene);
        glow.intensity = 1;

        const faultMaterial = new BABYLON.StandardMaterial(
            "faultMaterial",
            scene
        );

        faultMaterial.diffuseColor = new BABYLON.Color3(1, 0, 0);
        faultMaterial.emissiveColor = new BABYLON.Color3(1, 0, 0);

        const bearing1 = BABYLON.MeshBuilder.CreateSphere(
            "bearing-1",
            { diameter: largestSize * 0.025 },
            scene
        );

        const bearing2 = BABYLON.MeshBuilder.CreateSphere(
            "bearing-2",
            { diameter: largestSize * 0.025 },
            scene
        );

        bearing1.material = faultMaterial;
        bearing2.material = faultMaterial;

        const yellowGuardMeshes = modelMeshes.filter(mesh =>
            (mesh.name ?? "")
                .toLowerCase()
                .includes("yp-001.coupling guard235")
        );

        if (yellowGuardMeshes.length > 0) {
            let guardMin = new BABYLON.Vector3(
                Number.POSITIVE_INFINITY,
                Number.POSITIVE_INFINITY,
                Number.POSITIVE_INFINITY
            );

            let guardMax = new BABYLON.Vector3(
                Number.NEGATIVE_INFINITY,
                Number.NEGATIVE_INFINITY,
                Number.NEGATIVE_INFINITY
            );

            yellowGuardMeshes.forEach(mesh => {
                mesh.computeWorldMatrix(true);

                const box = mesh.getBoundingInfo().boundingBox;

                guardMin = BABYLON.Vector3.Minimize(
                    guardMin,
                    box.minimumWorld
                );

                guardMax = BABYLON.Vector3.Maximize(
                    guardMax,
                    box.maximumWorld
                );
            });

            const guardCenter = guardMin.add(guardMax).scale(0.5);
            const guardSize = guardMax.subtract(guardMin);
            const frontSurface =
                guardMax.z + largestSize * 0.012;

            bearing1.position = new BABYLON.Vector3(
                guardCenter.x - guardSize.x * 0.22,
                guardCenter.y,
                frontSurface
            );

            bearing2.position = new BABYLON.Vector3(
                guardCenter.x + guardSize.x * 0.22,
                guardCenter.y,
                frontSurface
            );
        } else {
            console.error("Yellow coupling guard mesh was not found.");

            bearing1.position = new BABYLON.Vector3(
                -0.0368,
                -0.0427,
                -0.63
            );

            bearing2.position = new BABYLON.Vector3(
                0.1392,
                -0.0427,
                -0.63
            );
        }

        bearing1.setEnabled(false);
        bearing2.setEnabled(false);

        [bearing1, bearing2].forEach(marker => {
            BABYLON.Animation.CreateAndStartAnimation(
                marker.name + "-pulse",
                marker,
                "scaling",
                30,
                30,
                new BABYLON.Vector3(0.8, 0.8, 0.8),
                new BABYLON.Vector3(1.3, 1.3, 1.3),
                BABYLON.Animation.ANIMATIONLOOPMODE_CYCLE
            );
        });

        const resizeHandler = () => engine.resize();
        window.addEventListener("resize", resizeHandler);

        engine.runRenderLoop(() => scene.render());

        this.viewers.set(canvasId, {
            engine,
            scene,
            resizeHandler,
            bearing1,
            bearing2
        });
    },

    setBearingFaults: function (
        canvasId,
        bearing1Fault,
        bearing2Fault
    ) {
        const viewer = this.viewers.get(canvasId);

        if (!viewer) {
            return;
        }

        viewer.bearing1.setEnabled(Boolean(bearing1Fault));
        viewer.bearing2.setEnabled(Boolean(bearing2Fault));
    },

    dispose: function (canvasId) {
        const viewer = this.viewers.get(canvasId);

        if (!viewer) {
            return;
        }

        window.removeEventListener(
            "resize",
            viewer.resizeHandler
        );

        viewer.scene.dispose();
        viewer.engine.dispose();
        this.viewers.delete(canvasId);
    }
};