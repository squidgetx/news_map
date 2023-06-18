// Pre-render d3 force-directed graph at server side
// Call node pre_render_d3_graph.js to generate d3_graph.html
// Original idea and framework borrowed from https://gist.github.com/mef/7044786

const d3 = require("d3"),
  jsdom = require("jsdom"),
  fs = require("fs"),
  assert = require("assert"),
  { ArgumentParser } = require("argparse"),
  htmlStub =
    '<html><head> \
	<style>.node { stroke: #fff; fill: #ccc; stroke-width: 1.5px; } \
	.link { stroke: #333; stroke-opacity: .5; stroke-width: 1.5px; }</style> \
	</head><body><div id="dataviz-container"></div></script></body></html>';

const { JSDOM } = jsdom;

/* 
Nodes: radius, id
Links: source, target, distance
return nodes
*/
const smwidth = 1200;
const smheight = 1200;

let getRadius = function (node) {
  if (node.hasOwnProperty("radius")) {
    return node.radius;
  }
  return Math.sqrt(node.size) * 2 + 2;
};

let getBasicSimulation = function (nodes, links) {
  return d3
    .forceSimulation(nodes)
    .force(
      "link",
      d3
        .forceLink(links)
        .id((d) => d.id)
        .distance((d) => d.distance * 2)
        .iterations(2)
        .strength((d) => 2 / d.weight)
    )
    .force("charge", d3.forceManyBody().strength(-1200));
};

let forceLayoutPreserveCollide = function (
  nodes,
  links,
  iterations,
  width,
  height
) {
  // Return an object with node.id => node
  // Strip out links that dont point to any node
  let nodeLookup = {};
  for (const n of nodes) {
    nodeLookup[n.id] = n;
  }
  let filtered_links = links.filter(
    (l) => l.source in nodeLookup && l.target in nodeLookup
  );
  let flstr = JSON.stringify(filtered_links);
  filtered_links = JSON.parse(flstr);

  for (const li in filtered_links) {
    let l = filtered_links[li];
    let r1 = getRadius(nodeLookup[l.source]);
    let r2 = getRadius(nodeLookup[l.target]);

    // figure out optimal distance:
    // if weight is closer to 1, distance should be r1+r2
    // if weight closer to 0, distance should be... further?
    //l.distance = (l.weight * (r1 + r2)) / 2;
    l.distance = (r1 + r2) / 2;
  }
  /*
  console.log(
    `Laying out ${nodes.length} nodes and ${links.length} links, with ${filtered_links.length} filtered links`
  );
  */

  let simulation = getBasicSimulation(nodes, filtered_links).force(
    "collision",
    d3.forceCollide().radius(getRadius)
  );

  for (let i = 0; i < iterations; i++) {
    simulation.tick();
  }
  return {
    nodes: Object.fromEntries(nodes.map((n) => [n.id, n])),
    links: filtered_links,
  };
};

let forceLayoutTerrible = function (nodes, links, iterations, width, height) {
  // Return an object with node.id => node
  // Strip out links that dont point to any node
  let nodeLookup = {};
  for (const n of nodes) {
    nodeLookup[n.id] = n;
  }
  let filtered_links = links.filter(
    (l) => l.source in nodeLookup && l.target in nodeLookup
  );
  let flstr = JSON.stringify(filtered_links);
  filtered_links = JSON.parse(flstr);

  for (const li in filtered_links) {
    let l = filtered_links[li];
    let r1 = getRadius(nodeLookup[l.source]);
    let r2 = getRadius(nodeLookup[l.target]);

    // figure out optimal distance:
    // if weight is closer to 1, distance should be r1+r2
    // if weight closer to 0, distance should be... further?
    l.distance = (r1 + r2) / 2;
  }
  /*
  console.log(
    `Laying out ${nodes.length} nodes and ${links.length} links, with ${filtered_links.length} filtered links`
  );
  */

  let simulation = d3
    .forceSimulation(nodes)
    .force(
      "link",
      d3
        .forceLink(filtered_links)
        .id((d) => d.id)
        .strength((d) => 0.01 / d.weight)
    )
    .force("charge", d3.forceManyBody().strength(-256))
    .force("center", d3.forceCenter(width / 2, height / 2));

  for (let i = 0; i < iterations / 2; i++) {
    simulation.tick();
  }
  return {
    nodes: Object.fromEntries(nodes.map((n) => [n.id, n])),
    links: filtered_links,
  };
};

let forceLayout = function (nodes, links, iterations, width, height) {
  // Return an object with node.id => node
  // Strip out links that dont point to any node
  let nodeLookup = {};
  for (const n of nodes) {
    nodeLookup[n.id] = n;
  }
  let filtered_links = links.filter(
    (l) => l.source in nodeLookup && l.target in nodeLookup
  );
  let flstr = JSON.stringify(filtered_links);
  filtered_links = JSON.parse(flstr);

  for (const li in filtered_links) {
    let l = filtered_links[li];
    let r1 = getRadius(nodeLookup[l.source]);
    let r2 = getRadius(nodeLookup[l.target]);

    // figure out optimal distance:
    // if weight is closer to 1, distance should be r1+r2
    // if weight closer to 0, distance should be... further?
    l.distance = (r1 + r2) / 2;
  }
  /*
  console.log(
    `Laying out ${nodes.length} nodes and ${links.length} links, with ${filtered_links.length} filtered links`
  );
  */

  let simulation = getBasicSimulation(nodes, filtered_links).force(
    "center",
    d3.forceCenter(width / 2, height / 2)
  );
  let simulationCollide = getBasicSimulation(nodes, filtered_links)
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force("collision", d3.forceCollide().radius(getRadius));

  //let simulationCollide = d3
  //// .forceSimulation(nodes)
  //.force("collision", d3.forceCollide().radius(getRadius));

  for (let i = 0; i < 20; i++) {
    //simulationCollide.tick();
  }

  for (let i = 0; i < iterations / 2; i++) {
    simulation.tick();
  }
  for (let i = 0; i < iterations / 2; i++) {
    simulationCollide.tick();
  }
  return {
    nodes: Object.fromEntries(nodes.map((n) => [n.id, n])),
    links: filtered_links,
  };
};

let getLinkedNodeIds = function (node_id, links) {
  return links
    .filter((l) => l.source == node_id || l.target == node_id)
    .map((l) => (l.source == node_id ? l.target : l.source));
};

let getUnitVec = function (vec1, vec2) {
  let dx = vec1.x - vec2.x;
  let dy = vec1.y - vec2.y;
  let mag = Math.sqrt(dx * dx + dy * dy);
  return { x: dx / mag, y: dy / mag };
};

let getCentroid = function (pts) {
  return {
    x: d3.mean(pts.map((p) => p.x)),
    y: d3.mean(pts.map((p) => p.y)),
  };
};

let layoutCluster = function (
  nodes,
  links,
  iterations,
  name,
  width,
  height,
  scale
) {
  // First, lay out the "core" nodes
  let coreNodes = nodes.filter((n) => n.type == "core" || n.type == "bridge");
  let coreLayout = forceLayout(coreNodes, links, iterations, width, height);
  if (coreNodes.length == 0) {
    coreLayout = forceLayout(nodes, links, iterations, width, height);
    write_layout(coreLayout, name, width, height);
    return nodes;
  }

  // Now, fix all the core node positions
  for (let node of coreNodes) {
    node.fx = coreLayout.nodes[node.id].x;
    node.fy = coreLayout.nodes[node.id].y;
  }

  // Calculate the center of mass of the core
  let centroid = getCentroid(coreNodes);
  /*

  for (let node of nodes.filter((n) => n.type == "bridge")) {
    // Don't initialize distance if pre-existing already exists
    if (node.hasOwnProperty("x") && node.hasOwnProperty("y")) {
      continue;
    }
    let linkNodeIds = getLinkedNodeIds(node.id, links);
    let linkNodes = linkNodeIds.map((id) => nodes.filter((n) => n.id == id)[0]);
    console.log(linkNodes);
    let ln = getCentroid(linkNodes);
    let dv = getUnitVec(ln, centroid);
    node.x = ln.x + dv.x * 1000;
    node.y = ln.y + dv.y * 1000;
    console.log(`Placing bridge node ${node.id} at ${node.x}, ${node.y}`);
  }*/

  // For all bridge node positions, set to distance
  for (let node of nodes.filter((n) => n.type == "leaf")) {
    // Don't initialize distance if pre-existing already exists
    if (node.hasOwnProperty("x") && node.hasOwnProperty("y")) {
      continue;
    }
    let linkNodeIds = getLinkedNodeIds(node.id, links);
    if (linkNodeIds.length == 1) {
      let linkNode = nodes.filter((n) => n.id == linkNodeIds[0])[0];
      //console.log(linkNode);
      let dv = getUnitVec(linkNode, centroid);
      let r = getRadius(node) + getRadius(linkNode);
      node.x = linkNode.x + dv.x * r;
      node.y = linkNode.y + dv.y * r;
      //console.log(`Placing leaf node ${node.id} at ${node.x}, ${node.y}`);
    } else if (linkNodeIds.length > 1) {
      console.log("uh oh");
    }
  }

  // Now, finish the force layout
  let fullLayout = forceLayoutPreserveCollide(
    nodes,
    links,
    1000,
    width,
    height
  );
  //console.log("centroid", centroid.x, centroid.y);

  write_layout(fullLayout, name, width, height, centroid);

  return nodes;
};

let write_layout = function (layout, name, width, height) {
  const { window } = new JSDOM(htmlStub).window;
  const EXPORT_SIZE = 1200;
  const nodes = Object.values(layout.nodes);
  const links = layout.links;
  // this callback function pre-renders the dataviz inside the html document,
  // then export result into a static html file
  /*
  console.log(
    `Writing ${name} with ${nodes.length} nodes and ${links.length} links`
  );
  */

  let el = window.document.querySelector("#dataviz-container");

  let xmin = Math.floor(d3.min(nodes, (n) => n.x - getRadius(n)));
  let xmax = Math.ceil(d3.max(nodes, (n) => n.x + getRadius(n)));
  let ymin = Math.floor(d3.min(nodes, (n) => n.y - getRadius(n)));
  let ymax = Math.ceil(d3.max(nodes, (n) => n.y + getRadius(n)));
  let vW = xmax - xmin;
  let vH = ymax - ymin;
  let vbox = `${xmin} ${ymin} ${vW} ${vH}`;
  let export_width = (EXPORT_SIZE * vW) / vH;
  let export_height = EXPORT_SIZE;
  if (vW > vH) {
    export_width = EXPORT_SIZE;
    export_height = (EXPORT_SIZE * vH) / vW;
  }
  // generate the graph
  let svg = d3
    .select(el)
    .append("svg")
    .attr("width", export_width)
    .attr("height", export_height)
    .attr("viewBox", vbox);

  const node = svg
    .append("g")
    .attr("stroke", "#fff")
    .attr("stroke-width", 1.5)
    .selectAll("circle")
    .data(nodes);

  const link = svg
    .append("g")
    .attr("stroke", "#111")
    .selectAll("line")
    .data(links);

  const linklines = link.join("line").attr("stroke-width", 1);
  const linklabels = link.join("text").attr("class", "linklabel");
  //.html((d) => `${d.weight.toFixed(2)}, ${d.distance.toFixed(2)}`);
  let nodeColor = {
    core: "coral",
    leaf: "cadetblue",
    bridge: "aquamarine",
  };
  const nodeCircles = node
    .join("circle")
    .attr("r", getRadius)
    .attr("fill", (d) => nodeColor[d.type] || "coral");
  const nodelabels = node.join("text").attr("style", "color: #aaa;");
  //.html((d) => `${d.id}, ${Math.floor(d.size)}, ${Math.floor(getRadius(d))}`);

  nodeCircles.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
  nodelabels.attr("x", (d) => d.x + 10).attr("y", (d) => d.y - 10);
  linklines
    .attr("x1", (d) => d.source.x)
    .attr("y1", (d) => d.source.y)
    .attr("x2", (d) => d.target.x)
    .attr("y2", (d) => d.target.y);
  linklabels
    .attr("x", (d) => (d.source.x + d.target.x) / 2)
    .attr("y", (d) => (d.source.y + d.target.y) / 2);

  window.document.documentElement.innerHTML += vbox;

  // save result in an html file
  fs.writeFile(
    `d3_graphs/${name}.html`,
    window.document.documentElement.innerHTML,
    function (err) {
      if (err) {
        console.log("error saving document", err);
      } else {
        //console.log(`d3_graph_${name}.html was saved!`);
      }
    }
  );
};

let mmc_force_layout_friction = function (
  metametaclusters,
  iterations,
  width,
  height
) {
  console.log("begin fl2");
  let mmc_nodes = metametaclusters["nodes"];
  let mmc_links = metametaclusters["edges"];
  let nodeLookup = {};
  for (const n of mmc_nodes) {
    nodeLookup[n.id] = n;
    n.sx = n.x;
    n.sy = n.y;
  }
  for (const li in mmc_links) {
    let l = mmc_links[li];
    let r1 = getRadius(nodeLookup[l.source]);
    let r2 = getRadius(nodeLookup[l.target]);
    l.distance = r1 + r2;
  }

  console.log(JSON.stringify(mmc_nodes[0]));
  let simulation = d3
    .forceSimulation(mmc_nodes)
    .force(
      "link",
      d3
        .forceLink(mmc_links)
        .id((d) => d.id)
        .distance((d) => d.distance / 4)
        .strength((d) => 1 / (d.weight + 1))
    )
    .force(
      "charge",
      d3.forceManyBody().strength((d) => -500)
    )
    .force(
      "radial",
      d3
        .forceRadial(
          1,
          (d) => d.sx,
          (d) => d.sy
        )
        .strength(0.9)
    );

  for (let i = 0; i < iterations / 2; i++) {
    simulation.tick();
  }
  console.log("begin sim 2");
  simulation = simulation.force(
    "collision",
    d3.forceCollide().radius(getRadius)
  );
  for (let i = 0; i < iterations / 2; i++) {
    simulation.tick();
  }
  let layout = {
    nodes: Object.fromEntries(mmc_nodes.map((n) => [n.id, n])),
    links: mmc_links,
  };
  console.log("write layout");
  write_layout(layout, "mmc2", width, height);
  for (let n of mmc_nodes) {
    if (isNaN(n.x)) {
      console.log(
        `${n.id} is NaN! radius is ${getRadius(n)} ${JSON.stringify(n)}`
      );
    }
  }
  return mmc_nodes;
};
let mmc_force_layout_2 = function (
  metametaclusters,
  iterations,
  width,
  height
) {
  console.log("begin fl2");
  let mmc_nodes = metametaclusters["nodes"];
  let mmc_links = metametaclusters["edges"];
  let nodeLookup = {};
  for (const n of mmc_nodes) {
    nodeLookup[n.id] = n;
  }
  let linkLookup = {};
  for (const li in mmc_links) {
    let l = mmc_links[li];
    let r1 = getRadius(nodeLookup[l.source]);
    let r2 = getRadius(nodeLookup[l.target]);
    if (l.source in linkLookup) {
      linkLookup[l.source].add(l.target);
    } else {
      linkLookup[l.source] = new Set();
      linkLookup[l.source].add(l.target);
    }
    if (l.target in linkLookup) {
      linkLookup[l.target].add(l.source);
    } else {
      linkLookup[l.target] = new Set();
      linkLookup[l.target].add(l.source);
    }
    l.distance = r1 + r2;
  }

  for (const n of mmc_nodes) {
    if (!n.hasOwnProperty("x")) {
      // start at a random point really far away
      n.x = width / 2 + Math.random() * 1000 - 500;
      n.y = height / 2 + Math.random() * 1000 - 500;
    }
  }
  console.log(JSON.stringify(mmc_links));
  console.log("begin sim", iterations);
  let simulation = d3
    .forceSimulation(mmc_nodes)
    .force(
      "link",
      d3
        .forceLink(mmc_links)
        .id((d) => d.id)
        .distance((d) => d.distance / 4)
        .strength((d) => 1 / (d.weight + 1))
      //.iterations(2)
    )
    .force(
      "charge",
      d3.forceManyBody().strength((d) => -2000)
    );

  for (let i = 0; i < iterations / 2; i++) {
    simulation.tick();
  }
  console.log("begin sim 2");
  simulation = simulation.force(
    "collision",
    d3.forceCollide().radius(getRadius)
  );
  for (let i = 0; i < iterations / 2; i++) {
    simulation.tick();
  }
  let layout = {
    nodes: Object.fromEntries(mmc_nodes.map((n) => [n.id, n])),
    links: mmc_links,
  };
  console.log("write layout");
  write_layout(layout, "mmc2", width, height);
  for (let n of mmc_nodes) {
    if (isNaN(n.x)) {
      console.log(
        `${n.id} is NaN! radius is ${getRadius(n)} ${JSON.stringify(n)}`
      );
    }
  }
  return mmc_nodes;
};

let mmc_force_layout = function (metametaclusters) {
  let mmc_nodes = metametaclusters["nodes"];

  console.log(mmc_nodes);
  let centers = [];
  const WIDTH = 9600;
  for (let i = 0; i < mmc_nodes.length; i++) {
    let center = mmc_nodes[i];
    if (!center.hasOwnProperty("x") || !center.hasOwnProperty("y")) {
      // initialize it using vector maths
      // get the Big Edges between this cluster, and any cluster that we've placed already
      // If this is the first cluster, just put it in the middle of the map
      // If there are more than one clusters that connect to this one that we've placed already,
      // I guess just put it in between them??
      if (i == 0) {
        //console.log("Placing at /2/2");
        center.x = WIDTH / 2;
        center.y = WIDTH / 2;
      } else {
        let mmc_links = getLinkedNodeIds(i, metametaclusters["edges"]).filter(
          (l) => l < i
        );
        //console.log(i, mmc_links);
        if (mmc_links.length == 1) {
          let mmc_id = mmc_links[0];
          // Get the center of the component we're going to link to, and
          // then we want to get the mean of the relevant bridge nodes
          let mmc_component_ids = new Set(layouts[mmc_id].map((n) => n.id));
          let i_component_ids = new Set(layouts[i].map((n) => n.id));
          let relevant_bridges = [];
          for (edge of big_edges) {
            if (
              mmc_component_ids.has(edge.source) &&
              i_component_ids.has(edge.target)
            ) {
              relevant_bridges.push(node_lookup[edge.source]);
            }
            if (
              mmc_component_ids.has(edge.target) &&
              i_component_ids.has(edge.source)
            ) {
              relevant_bridges.push(node_lookup[edge.target]);
            }
          }
          //console.log("relevant bridges:", relevant_bridges);
          let mmc_centroid = getCentroid(layouts[mmc_id]);
          let cx = d3.mean(relevant_bridges.map((n) => n.x));
          let cy = d3.mean(relevant_bridges.map((n) => n.y));
          let dv = getUnitVec({ x: cx, y: cy }, mmc_centroid);
          center.x = mmc_centroid.x + dv.x * 2000;
          center.y = mmc_centroid.y + dv.y * 2000;
          //console.log(`placing center at ${center.x}, ${center.y}`);
        } else {
          let xs = [];
          let ys = [];
          for (mmc_id of mmc_links) {
            xs.push(d3.mean(layouts[mmc_id].map((n) => n.x)));
            ys.push(d3.mean(layouts[mmc_id].map((n) => n.y)));
          }
          center.x = NaN; //d3.mean(xs);
          center.y = NaN; //d3.mean(ys);
          /*
          console.log(
            `placing center at ${center.x}, ${center.y} using median strategy`
          );
          */
        }
      }
    }
    centers.push(center);
  }
  return centers;
};

let mmc_grid_layout = function (metametaclusters) {
  let mmc_nodes = metametaclusters["nodes"];
  let centers = [];
  let GRID = 700;
  const WIDTH = 3600;
  let x = GRID / 2;
  let y = GRID / 2;
  for (let i = 0; i < mmc_nodes.length; i++) {
    let center = mmc_nodes[i];
    center.x = x;
    center.y = y;
    centers.push(center);
    x += GRID;
    if (x > WIDTH) {
      x = 0;
      y += GRID;
    }
  }
  return centers;
};

let apply_centers = function (layouts, centers) {
  for (let i = 0; i < layouts.length; i++) {
    let nodes = layouts[i];
    let centroid = getCentroid(nodes);
    let center = centers[i];
    for (node of nodes) {
      if (isNaN(node.x) || isNaN(node.y)) {
        node.x = center.x;
        node.y = center.y;
      } else {
        node.x += center.x - centroid.x;
        node.y += center.y - centroid.y;
      }
    }
  }
  return layouts;
};

// BEGIN MAIN

const parser = new ArgumentParser({
  description: "Layout script ",
});

parser.add_argument("-n", "--name", { type: "str", help: "name" });
parser.add_argument("-g", "--grid", {
  help: "use grid",
  action: "store_const",
  const: true,
  required: false,
});
parser.add_argument("-i", "--iter", {
  type: "int",
  help: "number of iterations",
  default: 1000,
});

let args = parser.parse_args();
let name = args.name;
let iterations = args.iter;
let metaclusters = JSON.parse(
  fs.readFileSync(`data/${name}.metaclusters.json`)
);
let metametaclusters = JSON.parse(
  fs.readFileSync(`data/${name}.metametaclusters.json`)
);
let big_edges = JSON.parse(fs.readFileSync(`data/${name}.fullgraph.json`));
let layouts = [];
console.log(metaclusters.length);
all_edges = [];
node_lookup = [];
distances = [];
for (let metacluster of metaclusters) {
  let nodes = layoutCluster(
    metacluster["nodes"],
    metacluster["edges"],
    1000,
    metacluster["id"],
    smwidth,
    smheight,
    1
  );

  all_edges = all_edges.concat(metacluster["edges"]);
  layouts.push(nodes);
  for (n of nodes) {
    node_lookup[n.id] = n;
  }
  let centroid = getCentroid(nodes);
  let dist = nodes.map(
    (n) =>
      Math.sqrt(
        (n.x - centroid.x) * (n.x - centroid.x) +
        (n.y - centroid.y) * (n.y - centroid.y)
      ) + getRadius(n)
  );
  distances.push(d3.max(dist));
}

for (n in metametaclusters["nodes"]) {
  metametaclusters["nodes"][n].radius = distances[n];
}

let mmc_centers;
if (args.grid) {
  mmc_centers = mmc_grid_layout(metametaclusters);
  layouts = apply_centers(layouts, mmc_centers);
  iterations = 0;
} else {
  //mmc_centers = mmc_force_layout_2(metametaclusters, 1000, 9600, 9600);
  mmc_centers = mmc_force_layout_friction(metametaclusters, 100, 9600, 9600);
  layouts = apply_centers(layouts, mmc_centers);
  iterations = 0;
}
layouts.flat().forEach((n) => {
  delete n.fx;
  delete n.fy;
});
let dummyLayout = forceLayout(layouts.flat(), all_edges, 0, 9600, 9600);
write_layout(dummyLayout, "all", 9600, 9600);
/*
// Reset nodes


//big_edges.forEach((n) => (n.weight = 5));
all_edges = all_edges.concat(big_edges);
let full_layout = forceLayout(
  all_nodes.filter((n) => n.type != "unconnected"),
  all_edges,
  iterations,
  9600,
  9600
);
write_layout(full_layout, "mmc", 9600, 9600);
for (let i = 0; i < layouts.length; i++) {
  let centroid = getCentroid(layouts[i]);
  mmc_centers[i] = centroid; //[centroid.x, centroid.y];
}
*/

fs.writeFile(
  `data/${name}.layout.json`,
  JSON.stringify({ layouts: layouts, centers: mmc_centers }),
  function (err) {
    if (err) {
      console.log("error saving document", err);
    } else {
      console.log(`data/${name}.layout.json saved!`);
    }
  }
);
