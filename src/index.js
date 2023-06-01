import * as d3 from "d3";
import svgPanZoom from "svg-pan-zoom";
import { getName, getEndDate } from "./files";

const SIZE = 800;
const SIDEBAR_WIDTH = 400;
const DEBUG = false;

let topics;
let topicSizes = [];
let topicMediaNames = [];
let date = "2020-02-01";
let interval = 28;
//let date = '2023-05-01'
//let interval = 26

let pois = [
  {
    "x": -2132.2188604824105,
    "y": -120.28837396163347,
    z: 3,
  },
  {
    "x": -4118.914625789113,
    "y": -669.3422385388845,
    z: 6.8,
  }, {
    "x": -9888.288576144943,
    "y": -6813.478495773421,
    z: 9.5
  },
  {
    "x": -3348.6783712431975,
    "y": -8255.50493513283,
    z: 9.0
  }
]

let poi_index = -1;

drawMaster();

function constrain(a, min, max) {
  if (a < min) {
    return min;
  }
  if (a > max) {
    return max;
  }
  return a;
}

function map(a, min_i, max_i, min_o, max_o) {
  return ((a - min_i) / (max_i - min_i)) * (max_o - min_o) + min_o;
}

function drawMaster() {
  let topicName = getName(date, interval, "topics.json");
  d3.json(topicName).then((topicData) => {
    topics = topicData;
    console.log(topics);
    topicSizes = [];
    for (let k in topics) {
      topicSizes[k] = topics[k]["size"];
      topicMediaNames[k] = []; //topics[k]["_metadata_"]["media_names"];
    }

    renderControlPanel(date, interval);
    //drawLayoutGraph();
    drawRegionsSVG();
  });
}

function renderControlPanel(date, interval) {
  let end_date = getEndDate(date, interval);
  const controls = document.getElementById("controls");
  const prevButton = getDateButton("Prev", true, prevMonth);
  const dateHeader = document.createElement("span");
  dateHeader.innerHTML = `Map generated for ${dateToStr(end_date)}`;
  dateHeader.className = "dateHeader";
  const nextButton = getDateButton("Next", true, nextMonth);
  controls.innerHTML = "";
  //controls.appendChild(prevButton);
  controls.appendChild(dateHeader);
  //controls.appendChild(nextButton);
}

function getDateButton(text, disabled, cb) {
  let btn = document.createElement("button");
  if (disabled) {
    btn.setAttribute("disabled", true);
  }
  btn.innerHTML = text;
  btn.onclick = cb;
  return btn;
}

function dateToStr(date) {
  let d = new Date(date);
  return d.toDateString();
}

function nextMonth() {
  let d = new Date(date);
  d.setUTCHours(d.getUTCHours() + 4);
  let nextDate = new Date(d);
  nextDate.setDate(d.getDate() + 7);
  date = nextDate.toISOString().slice(0, 10);
  drawMaster();
}

function prevMonth() {
  let d = new Date(date);
  d.setUTCHours(d.getUTCHours() + 4);
  let nextDate = new Date(d);
  nextDate.setDate(d.getDate() - 7);
  date = nextDate.toISOString().slice(0, 10);
  drawMaster();
}

const terrainGrid = [
  ["tundra", "taiga", "desert"], // dry
  ["taiga", "pasture", "savannah"], // medium
  ["snow", "forest", "rainforest"], // wet
];

function getTerrain(r) {
  let t = r.temperature;
  let m = r.moisture;
  let mIndex = Math.floor(m * 3);
  let tIndex = Math.floor(t * 3);

  return terrainGrid[mIndex][tIndex];
}

function getTerrainColorInterpolate(r) {
  let colors = [
    d3.interpolateRgb(getTerrainColor("desert"), getTerrainColor("savannah")),
    d3.interpolateRgb(getTerrainColor("savannah"), getTerrainColor("forest")),
    d3.interpolateRgb(getTerrainColor("forest"), getTerrainColor("rainforest")),
  ];
  let len = colors.length;
  let elev = constrain(r.elevation, 0, 0.999);
  let color = colors[Math.floor(elev * len)](
    elev * len - Math.floor(elev * len)
  );
  return color;
}

function getTerrainColor(t) {
  const TerrainColor = {
    tundra: "#fdfff0",
    rock: "#d4c9bc",
    desert: "#fff1cf",
    taiga: "#e1edd8",
    pasture: "#a3d4aa",
    savannah: "#dae39f",
    snow: "#ffffff",
    forest: "#216b2c",
    rainforest: "#185713",
    darksnow: "#f0e4d5",
  };
  return TerrainColor[t];
}

function drawRegionsSVG() {
  let renderRegionsSVG = function (region_data, vertex_data) {
    const selector = "#svgContainer";
    let svg = d3.select(selector).select("svg");
    if (svg.empty()) {
      svg = d3.select(selector).append("svg").attr("id", "map_main");
      let xExt = d3.extent(Object.values(vertex_data).map((r) => r.x));
      let yExt = d3.extent(Object.values(vertex_data).map((r) => r.y));
      let vBox = `${xExt[0]} ${yExt[0]} ${xExt[1] - xExt[0]} ${yExt[1] - yExt[0]
        }`;
      svg
        .attr("viewBox", vBox)
        .attr("width", window.innerWidth - SIDEBAR_WIDTH)
        .attr("height", window.innerHeight);
      svg.append("g").attr("class", "regions");
      svg.append("g").attr("class", "text-secondary");
      svg.append("g").attr("class", "text-primary");
      var defs = svg.append("defs");

      var filter = defs.append("filter").attr("id", "dropshadow");

      filter
        .append("feGaussianBlur")
        .attr("in", "SourceAlpha")
        .attr("stdDeviation", 0.0025)
        .attr("result", "blur");
      filter
        .append("feComponentTransfer")
        .attr("in", "blur")
        .attr("result", "offsetBlur")
        .html(
          '<feFuncA type="table" tableValues="0 .05 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"/>'
        );
      filter
        .append("feFlood")
        .attr("in", "offsetBlur")
        .attr("flood-color", "#fff")
        .attr("flood-opacity", "0.8")
        .attr("result", "offsetColor");
      filter
        .append("feComposite")
        .attr("in", "offsetColor")
        .attr("in2", "offsetBlur")
        .attr("operator", "in")
        .attr("result", "offsetBlur");

      var feMerge = filter.append("feMerge");
      feMerge.append("feMergeNode").attr("in", "offsetBlur");
      feMerge.append("feMergeNode").attr("in", "SourceGraphic");
    }
    let regions = svg.select(".regions").selectAll("path").data(region_data);
    regions
      .enter()
      .append("path")
      .merge(regions)
      .attr("d", (r) => {
        if (r.is_edge == "True") {
          return "";
        }
        let points = "M";
        for (const c of r.coordinates) {
          if (c != -1) {
            points += vertex_data[c].x + "," + vertex_data[c].y + " ";
          }
        }
        points = points.slice(0, -1);
        return points;
      })
      .style("fill", (r) => getTerrainColorInterpolate(r))
      .style("stroke", (r) => getTerrainColorInterpolate(r))
      .style("stroke-width", 0.0025)
      .style("cursor", "pointer")
      .attr("class", (r) => "t" + r.topic)
      .on("mouseover", (ev, r) => {
        d3.selectAll(`.t${r.topic}`)
          .style("fill", (r) => {
            let color = getTerrainColorInterpolate(r);
            return d3.color(color).darker();
          })
          .style("stroke", (r) => {
            let color = getTerrainColorInterpolate(r);
            return d3.color(color).darker();
          });
      })
      .on("mouseout", (ev, r) => {
        d3.selectAll(`.t${r.topic}`)
          .style("fill", (r) => {
            return getTerrainColorInterpolate(r);
          })
          .style("stroke", (r) => {
            return getTerrainColorInterpolate(r);
          });
      })
      .on("click", (_, r) => drawCard(r.topic));
    regions.exit().remove();

    let getFontSize = (t) => map(Math.sqrt(t.size), 0, 100, 0.005, 0.04);
    let drag = () => {
      function dragstarted(event, d) {
        d3.select(this).raise().style("cursor", "grabbed");
      }

      function dragged(event, d) {
        console.log(this);
        d3.select(this)
          .attr("x", (d.x = event.x))
          .attr("y", (d.y = event.y));
      }

      function dragended(event, d) {
        d3.select(this).attr("stroke", null);
      }

      return d3
        .drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended);
    };
    let labels = svg
      .select(".text-primary")
      .selectAll("text")
      .data(Object.values(topics))
      .join("text")
      .html((d) => d.region_name.toUpperCase())
      .style("fill", "#444")
      .style("text-anchor", "middle")
      .attr("x", (d) => d.x)
      .attr("y", (d) => d.y)
      .attr("filter", "url(#dropshadow)")
      .attr("font-size", (d) => `${getFontSize(d)}pt`)
      .attr("class", (d) => `f${Math.floor(getFontSize(d) * 1000)}`)
      .call(drag());

    return svg;
  };
  d3.tsv(getName(date, interval, "vertices.tsv")).then((vertexData) => {
    vertexData = Object.fromEntries(
      vertexData.map((d, i) => {
        return [
          d.index,
          {
            x: parseFloat(d.x),
            y: parseFloat(d.y),
          },
        ];
      })
    );
    d3.tsv(getName(date, interval, "regions.tsv")).then((regionData) => {
      console.log("regiondata", regionData);
      regionData = regionData.map((d, i) => {
        return {
          id: i,
          coordinates: JSON.parse(d.coordinates),
          elevation: parseFloat(d.elevation),
          flux: parseFloat(d.flux),
          moisture: constrain(
            parseFloat(d.moisture) * 3 * 0.4 + parseFloat(d.elevation) * 0.6,
            0,
            0.99
          ),
          temperature:
            parseFloat(d.temperature) * 0.2 +
            (1 - parseFloat(d.elevation)) * 0.8,
          topic: parseInt(d.topics),
          shadow: parseInt(d.shadow),
          is_edge: d.is_edge,
        };
      });
      console.log(Object.keys(vertexData).length, "vertexes");
      console.log(regionData.length, "triangles");
      let svg = renderRegionsSVG(regionData, vertexData);
      console.log("done constructing");
      let panControl = svgPanZoom("#map_main", {
        controlIconsEnabled: true,
        minZoom: 0.5,
        maxZoom: 12,
        onZoom: (zoom) => {
          console.log(zoom)
          /*
          const RANGE = 4;
          const MAX = 27;
          zoom = Math.floor(map(zoom, 1, 10, MAX, 0));
          for (let z = 0; z < zoom - RANGE; z++) {
            d3.selectAll(`.f${z}`).style("opacity", 0);
          }
          for (let z = zoom - RANGE; z < zoom + RANGE; z++) {
            d3.selectAll(`.f${z}`).style("opacity", 1);
            console.log("show .f", z);
            // Hide larger labels and show smaller ones
          }
          for (let z = zoom + RANGE; z < MAX; z++) {
            d3.selectAll(`.f${z}`).style("opacity", 0);
            // Hide larger labels and show smaller ones
          }
          */
        },
        onPan: (pan) => {
          console.log(pan)
        }
      });
      // code for manually jumping around for demo purposes
      document.getElementById('poi').addEventListener('click', () => {
        poi_index += 1
        if (poi_index >= pois.length) {
          poi_index = 0
        }
        panControl.zoom(pois[poi_index].z)
        panControl.pan({ x: pois[poi_index].x, y: pois[poi_index].y })




      })
    });





  });
}

let handleMouseOver = function (ev, d) {
  d3.select(this).transition().duration(0.1).attr("r", getSize(1.1));
  drawCard(d.id, d);
};

let drawCard = function (topic_id) {
  document.getElementById("placename").innerHTML =
    topics[topic_id]["region_name"];

  let articles = topics[topic_id]["articles"];
  renderArticles(articles);
  if (DEBUG) {
    renderDebugContent(topic_id);
  }
};

let renderDebugContent = function (topic_id) {
  let relevant_words = topics[topic_id]["relevant_words"]
    .filter((a) => a[1] > 0)
    .sort(function (a, b) {
      return b[1] - a[1];
    })
    .map(
      (word_freq) =>
        `<p class='word' style='font-size:${Math.sqrt(word_freq[1]) + 6}pt'>${word_freq[0]
        }</p>`
    )
    .join("");
  let common_words = topics[topic_id]["common_words"]
    .filter((a) => a[1] > 0)
    .sort(function (a, b) {
      return b[1] - a[1];
    })
    .map(
      (word_freq) =>
        `<p class='word' style='font-size:${Math.sqrt(word_freq[1]) + 6}pt'>${word_freq[0]
        }</p>`
    )
    .join("");

  document.getElementById(
    "debug"
  ).innerHTML = `<p>Size: ${topics[topic_id].size}</p>`;
  document.getElementById("debug").innerHTML += "<h3>Most Relevant Words</h3>";
  document.getElementById("debug").innerHTML += relevant_words;
  document.getElementById("debug").innerHTML += "<h3>Most Common Words</h3>";
  document.getElementById("debug").innerHTML += common_words;
};

let renderArticles = function (articles) {
  let articleList = document.getElementById("articleList");
  let articleNode = document.createElement("div");
  for (const article of articles) {
    let linkp = document.createElement("p");
    let link = document.createElement("a");
    link.href = article.url;
    link.className = "articleLink";
    link.innerHTML = article.title;
    linkp.appendChild(link);
    let pDate = document.createElement("span");
    pDate.innerHTML = dateToStr(article.publish_date);
    pDate.className = "articleDate";
    linkp.appendChild(pDate);
    articleNode.appendChild(linkp);
  }
  articleList.replaceChildren(articleNode);
};

let handleMouseOut = function (ev, d) {
  d3.select(this).transition().duration(0.1).attr("r", getSize(1));
};

let getSize = function (mult) {
  return (d) => Math.sqrt(topicSizes[d.id]) * 2 * mult;
};

function drawLayoutGraph() {
  let gcolor = (function () {
    const scale = d3.scaleOrdinal(d3.schemeCategory10);
    return (d) => scale(parseInt(d.group));
  })();

  let graph = function (data) {
    const nodes = data.layouts
      .map((layout, i) =>
        layout.map((d) => {
          return {
            group: i,
            id: parseInt(d.id),
            x: parseFloat(d.x),
            y: parseFloat(d.y),
          };
        })
      )
      .flat();
    console.log("layout", nodes);
    let xExt = d3.extent(nodes.map((n) => n.x));
    let yExt = d3.extent(nodes.map((n) => n.y));
    let maxRadius = d3.max(nodes.map(getSize(1)));
    console.log(yExt);

    const svg = d3
      .create("svg")
      .attr("class", "layoutSVG")
      /*
      .attr("viewBox", [
        xExt[0] - maxRadius,
        yExt[0] - maxRadius,
        xExt[1] - xExt[0] + maxRadius * 2,
        yExt[1] - yExt[0] + maxRadius * 2,
      ]);
      */
      .attr("viewBox", [0, 0, 9600, 9600]);

    const node = svg
      .append("g")
      .selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", getSize(1))
      .attr("cx", (d) => d.x)
      .attr("cy", (d) => d.y)
      .attr("topic", (d) => d.id)
      .attr("fill", gcolor)
      .on("mouseover", handleMouseOver)
      .on("mouseout", handleMouseOut);

    return svg.node();
  };
  let jsonName = getName(date, interval, "layout.json");
  d3.json(jsonName).then((data) => {
    document.getElementById("layoutContainer").replaceChildren(graph(data));
  });
}
