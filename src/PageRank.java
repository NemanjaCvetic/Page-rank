import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.*;

public class PageRank {
    private final int numVertices;
    private final int numEdges;
    private final double dampingFactor;
    private final int maxIterations;
    private final List<List<Integer>> adjacencyList;
    private final Random random;
    private final int availableProcessors;
    private double[] pageRanks;

    public enum ExecutionMode {
        SEQUENTIAL,
        PARALLEL,
        DISTRIBUTED
    }

    public PageRank(int numVertices, int numEdges, double dampingFactor, int maxIterations, long seed) {
        this.numVertices = numVertices;
        this.numEdges = numEdges;
        this.dampingFactor = dampingFactor;
        this.maxIterations = maxIterations;
        this.random = new Random(seed);
        this.adjacencyList = new ArrayList<>(numVertices);
        this.availableProcessors = Runtime.getRuntime().availableProcessors();

        // Initialize adjacency list
        for (int i = 0; i < numVertices; i++) {
            adjacencyList.add(new ArrayList<>());
        }

        // Generate random graph
        generateRandomGraph();

        // Initialize PageRank scores
        this.pageRanks = new double[numVertices];
        Arrays.fill(pageRanks, 1.0 / numVertices);
    }

    private void generateRandomGraph() {
        int edgesCreated = 0;
        while (edgesCreated < numEdges) {
            int from = random.nextInt(numVertices);
            int to = random.nextInt(numVertices);

            // Avoid self-loops and duplicate edges
            if (from != to && !adjacencyList.get(from).contains(to)) {
                adjacencyList.get(from).add(to);
                edgesCreated++;
            }
        }
    }

    public void computePageRank(ExecutionMode mode) {
        long startTime = System.currentTimeMillis();

        switch (mode) {
            case SEQUENTIAL:
                computeSequential();
                break;
            case PARALLEL:
                computeParallel();
                break;
            case DISTRIBUTED:
                computeDistributed();
                break;
        }

        long endTime = System.currentTimeMillis();
        System.out.println("Total computation time: " + (endTime - startTime) + "ms");
    }

    private void computeSequential() {
        for (int iteration = 0; iteration < maxIterations; iteration++) {
            long iterationStart = System.currentTimeMillis();
            double[] newPageRanks = new double[numVertices];

            // Initialize with damping factor
            Arrays.fill(newPageRanks, (1 - dampingFactor) / numVertices);

            // Calculate new PageRank for each vertex
            for (int i = 0; i < numVertices; i++) {
                for (int outNeighbor : adjacencyList.get(i)) {
                    newPageRanks[outNeighbor] += dampingFactor * pageRanks[i] / adjacencyList.get(i).size();
                }
            }

            pageRanks = newPageRanks;
            long iterationEnd = System.currentTimeMillis();
            System.out.println("Iteration " + (iteration + 1) + " time: " + (iterationEnd - iterationStart) + "ms");
        }
    }

    private void computeParallel() {
        ForkJoinPool pool = new ForkJoinPool(availableProcessors);

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            long iterationStart = System.currentTimeMillis();
            double[] newPageRanks = new double[numVertices];
            Arrays.fill(newPageRanks, (1 - dampingFactor) / numVertices);

            pool.submit(() ->
                    IntStream.range(0, numVertices).parallel().forEach(i -> {
                        for (int outNeighbor : adjacencyList.get(i)) {
                            synchronized (newPageRanks) {
                                newPageRanks[outNeighbor] += dampingFactor * pageRanks[i] / adjacencyList.get(i).size();
                            }
                        }
                    })
            ).join();

            pageRanks = newPageRanks;
            long iterationEnd = System.currentTimeMillis();
            System.out.println("Iteration " + (iteration + 1) + " time: " + (iterationEnd - iterationStart) + "ms");
        }
    }

    private void computeDistributed() {
        // Placeholder for distributed computation
        // This would typically involve network communication and data partitioning
        System.out.println("Distributed computation not implemented yet");
    }

    public void saveGraphToCSV(String filename) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("source,target");
            for (int i = 0; i < numVertices; i++) {
                for (int target : adjacencyList.get(i)) {
                    writer.println(i + "," + target);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void savePageRanksToCSV(String filename) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            writer.println("vertex,score");
            for (int i = 0; i < numVertices; i++) {
                writer.println(i + "," + pageRanks[i]);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        // Example usage
        int numVertices = 1000;
        int numEdges = 5000;
        double dampingFactor = 0.85;
        int maxIterations = 100;
        long seed = 42L;

        PageRank pageRank = new PageRank(numVertices, numEdges, dampingFactor, maxIterations, seed);

        // Run sequential version
        System.out.println("Running sequential computation...");
        pageRank.computePageRank(ExecutionMode.SEQUENTIAL);
        pageRank.saveGraphToCSV("graph.csv");
        pageRank.savePageRanksToCSV("pageranks.csv");

        // Run parallel version
        System.out.println("\nRunning parallel computation...");
        pageRank.computePageRank(ExecutionMode.PARALLEL);
    }
}