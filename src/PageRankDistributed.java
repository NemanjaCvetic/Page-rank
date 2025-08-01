import java.io.*;
import java.util.*;
import mpi.*;

public class PageRankDistributed {
    private final int numVertices;
    private final int numEdges;
    private final double dampingFactor;
    private final int maxIterations;
    private final List<List<Integer>> adjacencyList;
    private final Random random;
    private double[] pageRanks;
    private double[] outDegrees;

    // MPI-specific variables
    private int mpiRank = 0;
    private int mpiSize = 1;
    private int startVertex = 0;
    private int endVertex = 0;

    public PageRankDistributed(int numVertices, int numEdges, double dampingFactor, int maxIterations, long seed) {
        this.numVertices = numVertices;
        this.numEdges = numEdges;
        this.dampingFactor = dampingFactor;
        this.maxIterations = maxIterations;
        this.random = new Random(seed);
        this.adjacencyList = new ArrayList<>(numVertices);
        this.outDegrees = new double[numVertices];

        // Initialize adjacency list
        for (int i = 0; i < numVertices; i++) {
            adjacencyList.add(new ArrayList<>());
        }


        generateRandomGraph();
        calculateOutDegrees();

        // Initialize PageRank scores
        this.pageRanks = new double[numVertices];
        Arrays.fill(pageRanks, 1.0 / numVertices);
    }

    private void generateRandomGraph() {
        Set<String> edges = new HashSet<>();
        int edgesCreated = 0;

        while (edgesCreated < numEdges) {
            int from = random.nextInt(numVertices);
            int to = random.nextInt(numVertices);
            String edge = from + "-" + to;

            if (from != to && !edges.contains(edge)) {
                adjacencyList.get(from).add(to);
                edges.add(edge);
                edgesCreated++;
            }
        }
    }

    private void calculateOutDegrees() {
        for (int i = 0; i < numVertices; i++) {
            outDegrees[i] = adjacencyList.get(i).size();
            if (outDegrees[i] == 0) {
                outDegrees[i] = 1; // Handle dangling nodes
            }
        }
    }

    public void initializeMPI() {
        try {
            // Check if MPI is available before initializing
            if (!isMPIAvailable()) {
                System.out.println("MPI not available, using single process mode");
                this.mpiRank = 0;
                this.mpiSize = 1;
                this.startVertex = 0;
                this.endVertex = numVertices;
                return;
            }

            // MPI initialization
            this.mpiRank = MPI.COMM_WORLD.Rank();
            this.mpiSize = MPI.COMM_WORLD.Size();

            // Distribute vertices across processes
            int verticesPerProcess = numVertices / mpiSize;
            this.startVertex = mpiRank * verticesPerProcess;
            this.endVertex = (mpiRank == mpiSize - 1) ? numVertices : startVertex + verticesPerProcess;

            if (mpiRank == 0) {
                System.out.println("MPI initialized with " + mpiSize + " processes");
                System.out.println("Total vertices: " + numVertices);
                System.out.println("Vertices per process: " + verticesPerProcess);
            }

            System.out.println("Process " + mpiRank + " handles vertices " + startVertex + " to " + (endVertex - 1));

            // Synchronize all processes
            MPI.COMM_WORLD.Barrier();

        } catch (Exception e) {
            System.err.println("Error initializing MPI on process " + mpiRank + ": " + e.getMessage());
            e.printStackTrace();
            // Fallback to single process
            this.mpiRank = 0;
            this.mpiSize = 1;
            this.startVertex = 0;
            this.endVertex = numVertices;
        }
    }

    private boolean isMPIAvailable() {
        try {
            // Try to access MPI functionality
            MPI.COMM_WORLD.Rank();
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    public void computePageRank() {
        long startTime = System.currentTimeMillis();

        try {
            // Check if MPI is properly initialized
            if (!isMPIAvailable()) {
                if (mpiRank == 0) {
                    System.out.println("MPI not available or not properly initialized");
                    System.out.println("Cannot run distributed computation");
                }
                return;
            }

            if (mpiRank == 0) {
                System.out.println("Running distributed PageRank on " + mpiSize + " processes");
                System.out.println("Processing " + numVertices + " vertices, " + numEdges + " edges");
            }



            for (int iteration = 0; iteration < maxIterations; iteration++) {
                long iterationStart = System.currentTimeMillis();

                // Create local contributions array
                double[] localContributions = new double[numVertices];

                // Each process computes contributions for its assigned vertices
                computeLocalContributions(localContributions);

                // Sum all contributions across processes
                double[] globalContributions = allReduceContributions(localContributions);

                // Update PageRank values
                for (int i = 0; i < numVertices; i++) {
                    pageRanks[i] = (1 - dampingFactor) / numVertices + globalContributions[i];
                }

                if (mpiRank == 0) {
                    long iterationEnd = System.currentTimeMillis();
                    System.out.println("Iteration " + (iteration + 1) + " time: " +
                            (iterationEnd - iterationStart) + "ms");
                }
            }
        } catch (Exception e) {
            System.err.println("Error in distributed computation: " + e.getMessage());
            e.printStackTrace();
        }

        if (mpiRank == 0) {
            long endTime = System.currentTimeMillis();
            System.out.println("Total computation time: " + (endTime - startTime) + "ms");
        }
    }



    private void computeLocalContributions(double[] localContributions) {
        // Each process computes contributions for its assigned vertices
        for (int i = startVertex; i < endVertex; i++) {
            if (outDegrees[i] > 0) {
                double contribution = dampingFactor * pageRanks[i] / outDegrees[i];
                for (int outNeighbor : adjacencyList.get(i)) {
                    localContributions[outNeighbor] += contribution;
                }
            }
        }
    }

    private double[] allReduceContributions(double[] localContributions) {
        double[] globalContributions = new double[numVertices];

        try {
            // MPI AllReduce operation
            MPI.COMM_WORLD.Allreduce(localContributions, 0, globalContributions, 0,
                    numVertices, MPI.DOUBLE, MPI.SUM);
        } catch (MPIException e) {
            System.err.println("MPI AllReduce failed: " + e.getMessage());
            // Fallback: just use local contributions
            System.arraycopy(localContributions, 0, globalContributions, 0, numVertices);
        }

        return globalContributions;
    }

    public void saveGraphToCSV(String filename) {
        if (mpiRank == 0) {
            try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
                writer.println("source,target");
                for (int i = 0; i < numVertices; i++) {
                    for (int target : adjacencyList.get(i)) {
                        writer.println(i + "," + target);
                    }
                }
                System.out.println("Graph saved to " + filename);
            } catch (IOException e) {
                System.err.println("Error saving graph: " + e.getMessage());
            }
        }
    }

    public void savePageRanksToCSV(String filename) {
        if (mpiRank == 0) {
            try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
                writer.println("vertex,score");
                for (int i = 0; i < numVertices; i++) {
                    writer.println(i + "," + pageRanks[i]);
                }
                System.out.println("PageRank scores saved to " + filename);
            } catch (IOException e) {
                System.err.println("Error saving PageRank scores: " + e.getMessage());
            }
        }
    }

    public void printTopVertices(int top) {
        if (mpiRank == 0) {
            List<VertexScore> vertexScores = new ArrayList<>();
            for (int i = 0; i < numVertices; i++) {
                vertexScores.add(new VertexScore(i, pageRanks[i]));
            }

            vertexScores.sort((a, b) -> Double.compare(b.score, a.score));

            System.out.println("\nTop " + top + " vertices by PageRank score:");
            for (int i = 0; i < Math.min(top, vertexScores.size()); i++) {
                VertexScore vs = vertexScores.get(i);
                System.out.printf("Vertex %d: %.6f\n", vs.vertex, vs.score);
            }
        }
    }

    private static class VertexScore {
        int vertex;
        double score;

        VertexScore(int vertex, double score) {
            this.vertex = vertex;
            this.score = score;
        }
    }

    public static void main(String[] args) {
        boolean mpiInitialized = false;

        try {
            // Try to initialize MPI
            MPI.Init(args);
            mpiInitialized = true;
        } catch (Exception e) {
            System.out.println("MPI initialization failed or not available: " + e.getMessage());
            System.out.println("Cannot run distributed PageRank without MPI");
            return;
        }

        // Filter out MPJ Express arguments based on position and known patterns
        List<String> userArgs = new ArrayList<>();

        // MPJ Express typically adds: [rank] [size] [device] [user_args...]
        // So we skip the first few arguments that match MPJ patterns
        boolean foundUserArgs = false;
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];

            // Skip MPJ internal arguments at the beginning
            if (!foundUserArgs) {
                // Skip single digits (0-9) which are typically rank/size
                if (arg.matches("^[0-9]$")) {
                    continue;
                }
                // Skip device names
                if (arg.equals("smpdev") || arg.equals("niodev") || arg.equals("mxdev")) {
                    continue;
                }
                // If we find a larger number or other argument, start collecting user args
                foundUserArgs = true;
            }

            // Collect user arguments
            if (foundUserArgs) {
                userArgs.add(arg);
            }
        }

        // Debug: Print filtered arguments
        if (MPI.COMM_WORLD.Rank() == 0) {
            System.out.println("Filtered arguments: " + userArgs.size());
            for (int i = 0; i < userArgs.size(); i++) {
                System.out.println("User Argument " + i + ": '" + userArgs.get(i) + "'");
            }
        }

        if (userArgs.size() < 4) {
            if (MPI.COMM_WORLD.Rank() == 0) {
                System.out.println("Usage: mpjrun.sh -np <processes> PageRankDistributed <vertices> <edges> <damping_factor> <max_iterations> [seed]");
                System.out.println("Example: mpjrun.sh -np 4 PageRankDistributed 1000 5000 0.85 100 42");
            }
            if (mpiInitialized) {
                try {
                    MPI.Finalize();
                } catch (Exception e) {
                    // Ignore finalization errors
                }
            }
            return;
        }

        try {
            int numVertices = Integer.parseInt(userArgs.get(0));
            int numEdges = Integer.parseInt(userArgs.get(1));
            double dampingFactor = Double.parseDouble(userArgs.get(2));
            int maxIterations = Integer.parseInt(userArgs.get(3));
            long seed = userArgs.size() > 4 ? Long.parseLong(userArgs.get(4)) : 42L;

            PageRankDistributed pageRank = new PageRankDistributed(numVertices, numEdges, dampingFactor, maxIterations, seed);

            // Initialize MPI settings
            pageRank.initializeMPI();

            if (pageRank.mpiRank == 0) {
                System.out.println("=== Distributed PageRank Configuration ===");
                System.out.println("Vertices: " + numVertices);
                System.out.println("Edges: " + numEdges);
                System.out.println("Damping Factor: " + dampingFactor);
                System.out.println("Max Iterations: " + maxIterations);
                System.out.println("Seed: " + seed);
                System.out.println("MPI Processes: " + pageRank.mpiSize);
                System.out.println("=========================================\n");
            }


            pageRank.computePageRank();

            // Save results (only from rank 0)
            pageRank.saveGraphToCSV("graph_distributed.csv");
            pageRank.savePageRanksToCSV("pageranks_distributed.csv");
            pageRank.printTopVertices(10);

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        } finally {
            if (mpiInitialized) {
                try {
                    MPI.Finalize();
                } catch (Exception e) {
                    System.err.println("Error finalizing MPI: " + e.getMessage());
                }
            }
        }
    }
}