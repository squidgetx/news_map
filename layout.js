// Pre-render d3 force-directed graph at server side
// Call node pre_render_d3_graph.js to generate d3_graph.html
// Original idea and framework borrowed from https://gist.github.com/mef/7044786

const d3 = require("d3"),
  jsdom = require("jsdom"),
  fs = require("fs"),
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

let layout = function (
  nodes,
  links,
  iterations,
  name,
  width,
  height,
  scale,
  collide
) {
  const { window } = new JSDOM(htmlStub).window;
  // this callback function pre-renders the dataviz inside the html document,
  // then export result into a static html file

  let el = window.document.querySelector("#dataviz-container"),
    body = window.document.querySelector("body");

  // generate the graph
  let svg = d3
    .select(el)
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  let nodeLookup = {};
  for (const n of nodes) {
    nodeLookup[n.id] = n;
  }
  for (const li in links) {
    let l = links[li];
    let r1 = getRadius(nodeLookup[l.source]);
    let r2 = getRadius(nodeLookup[l.target]);

    // figure out optimal distance:
    // if weight is closer to 1, distance should be r1+r2
    // if weight closer to 0, distance should be... further?
    l.distance = (l.weight * (r1 + r2)) / 2;
  }
  let linkForce = d3
    .forceLink(links)
    .id((d) => d.id)
    .distance((d) => d.distance);

  if (name == "mmc") {
    linkForce = linkForce.strength(2.1);
  }

  var simulation = d3
    .forceSimulation(nodes)
    .force("link", linkForce)
    .force(
      "charge",
      d3.forceManyBody().strength((d) => -256 * scale)
    )
    .force("center", d3.forceCenter(width / 2, height / 2))
    .on("tick", ticked);

  const node = svg
    .append("g")
    .attr("stroke", "#fff")
    .attr("stroke-width", 1.5)
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("r", getRadius)
    .attr("fill", "coral");
  const link = svg
    .append("g")
    .attr("stroke", "#111")
    .selectAll("line")
    .data(links);

  const linklines = link.join("line").attr("stroke-width", 4);
  const linklabels = link
    .join("text")
    .attr("class", "linklabel")
    .html((d) => d.distance.toFixed(2));
  debugger;
  function ticked() {}

  // Here is the key. Without calling force.tick(), the simulation will not start and the nodes and links
  // will not have coordinates.
  for (let i = 0; i < iterations * 0.75; i++) {
    simulation.tick();
  }
  simulation.force("collision", d3.forceCollide().radius(getRadius));
  for (let i = 0; i < iterations * 0.25; i++) {
    simulation.tick();
  }

  node
    .data(nodes)
    .attr("cx", (d) => d.x)
    .attr("cy", (d) => d.y);
  linklines
    .attr("x1", (d) => d.source.x)
    .attr("y1", (d) => d.source.y)
    .attr("x2", (d) => d.target.x)
    .attr("y2", (d) => d.target.y);
  linklabels
    .attr("x", (d) => (d.source.x + d.target.x) / 2)
    .attr("y", (d) => (d.source.y + d.target.y) / 2);
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

  return nodes;
};

const parser = new ArgumentParser({
  description: "Layout script ",
});

parser.add_argument("-n", "--name", { type: "str", help: "name" });
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
let layouts = [];
let metacluster_sizes = {};
console.log(metaclusters.length);
debugger;
for (let metacluster of metaclusters) {
  let nodes = layout(
    metacluster["nodes"],
    metacluster["edges"],
    iterations,
    metacluster["id"],
    smwidth,
    smheight,
    1
  );
  layouts.push(nodes);
  distances = nodes.map((n) =>
    Math.sqrt(
      (n.x - smwidth / 2) * (n.x - smwidth / 2) +
        (n.y - smheight / 2) * (n.y - smheight / 2)
    )
  );
  metacluster_sizes[metacluster["id"]] = d3.max(distances);
  let max_index = d3.maxIndex(distances);
  metacluster_sizes[metacluster["id"]]; // += nodes.map(getRadius)[max_index];
}

for (let node of metametaclusters["nodes"]) {
  node.radius = metacluster_sizes[node["id"]];
}

let mmc_layout = layout(
  metametaclusters["nodes"],
  metametaclusters["edges"],
  iterations,
  "mmc",
  9600,
  9600,
  12
);

let centers = [];
let x = metacluster_sizes[0] / 2;
let y = metacluster_sizes[0] / 2;
let maxy = 0;
for (let i = 0; i < mmc_layout.length; i++) {
  let nodes = layouts[i];
  let center = mmc_layout[i];

  centers.push(center);
  if (false) {
    center.x = x;
    center.y = y;
  }
  const mmc_scale = 1;
  for (node of nodes) {
    node.x += center.x * mmc_scale;
    node.y += center.y * mmc_scale;
  }
  x += metacluster_sizes[i] * 2;
  if (metacluster_sizes[i] * 2 > maxy) {
    maxy = metacluster_sizes[i] * 2;
  }
  if (x > 1600) {
    x = 0;
    y += maxy;
    maxy = 0;
  }
}
debugger;

fs.writeFile(
  `data/${name}.layout.json`,
  JSON.stringify({ layouts: layouts, centers: centers }),
  function (err) {
    if (err) {
      console.log("error saving document", err);
    } else {
      console.log(`data/${name}.layout.json saved!`);
    }
  }
);
