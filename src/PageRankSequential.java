import java.io.*;
import java.util.*;

public class PageRankSequential {
    private final int numVertices;
    private final int numEdges;
    private final double dampingFactor;
    private final int maxIterations;
    private final List<List<Integer>> adjacencyList;
    private final Random random;
    private double[] pageRanks;
    private double[] outDegrees;

    public PageRankSequential(int numVertices, int numEdges, double dampingFactor, int maxIterations, long seed) {
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

        // Generate random graph
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

    public void computePageRank() {
        long startTime = System.currentTimeMillis();

        System.out.println("Running sequential PageRank on " + numVertices + " vertices, " + numEdges + " edges");

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            long iterationStart = System.currentTimeMillis();
            double[] newPageRanks = new double[numVertices];
            Arrays.fill(newPageRanks, (1 - dampingFactor) / numVertices);

            for (int i = 0; i < numVertices; i++) {
                if (outDegrees[i] > 0) {
                    double contribution = dampingFactor * pageRanks[i] / outDegrees[i];
                    for (int outNeighbor : adjacencyList.get(i)) {
                        newPageRanks[outNeighbor] += contribution;
                    }
                }
            }

            pageRanks = newPageRanks;

            long iterationEnd = System.currentTimeMillis();
            System.out.println("Iteration " + (iteration + 1) + " time: " + (iterationEnd - iterationStart) + "ms");
        }

        long endTime = System.currentTimeMillis();
        System.out.println("Total computation time: " + (endTime - startTime) + "ms");
    }

    public void saveGraphToCSV(String filename) {
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

    public void savePageRanksToCSV(String filename) {
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

    public void printTopVertices(int top) {
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

    private static class VertexScore {
        int vertex;
        double score;

        VertexScore(int vertex, double score) {
            this.vertex = vertex;
            this.score = score;
        }
    }

    public static void main(String[] args) {
        if (args.length < 4) {
            System.out.println("Usage: java PageRankSequential <vertices> <edges> <damping_factor> <max_iterations> [seed]");
            System.out.println("Example: java PageRankSequential 1000 5000 0.85 100 42");
            return;
        }

        try {
            int numVertices = Integer.parseInt(args[0]);
            int numEdges = Integer.parseInt(args[1]);
            double dampingFactor = Double.parseDouble(args[2]);
            int maxIterations = Integer.parseInt(args[3]);
            long seed = args.length > 4 ? Long.parseLong(args[4]) : 42L;

            System.out.println("=== Sequential PageRank Configuration ===");
            System.out.println("Vertices: " + numVertices);
            System.out.println("Edges: " + numEdges);
            System.out.println("Damping Factor: " + dampingFactor);
            System.out.println("Max Iterations: " + maxIterations);
            System.out.println("Seed: " + seed);
            System.out.println("=======================================\n");

            PageRankSequential pageRank = new PageRankSequential(numVertices, numEdges, dampingFactor, maxIterations, seed);

            // Run computation
            pageRank.computePageRank();

            // Save results
            pageRank.saveGraphToCSV("graph_sequential.csv");
            pageRank.savePageRanksToCSV("pageranks_sequential.csv");
            pageRank.printTopVertices(10);

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}