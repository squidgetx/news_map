const SIZE = 800;
const SIDEBAR_WIDTH = 400;
const DEBUG = false;

// Initialize the Amazon Cognito credentials provider
const S3_BUCKET = "fogofwar";
var s3 = new AWS.S3({
  apiVersion: "2006-03-01",
  params: { Bucket: albumBucketName },
});

AWS.config.region = "us-east-2"; // Region
AWS.config.credentials = new AWS.CognitoIdentityCredentials({
  IdentityPoolId: "us-east-2:f4537342-bcf0-45e1-b055-b95ee3752b71",
});

let vertex_data, region_data, tsne, topics;
let topicSizes = (topicMediaNames = []);
let mode = 0;
let date = "2020-02-01";
let end_date = "";
let interval = 28;
const DAY_MS = 1000 * 60 * 60 * 24;
const TOPIC = "us_mainstream_stories_trunc";

function getName(date, interval, suffix) {
  let start = new Date(date);
  start.setUTCHours(start.getUTCHours() + 4);
  let end = new Date(start);
  end.setDate(start.getDate() + interval);
  let startStr = start.toISOString().slice(0, 10);
  let endStr = end.toISOString().slice(0, 10);
  end_date = endStr;
  let str = `data/${TOPIC}_${startStr}_${endStr}.${suffix}`;
  console.log(str);
  return str;
}

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
    for (k in topics) {
      topicSizes[k] = topics[k]["size"];
      topicMediaNames[k] = []; //topics[k]["_metadata_"]["media_names"];
    }

    renderControlPanel(date, end_date);
    //drawLayoutGraph();
    drawRegionsSVG();
  });
}

function renderControlPanel(date, end_date) {
  console.log(end_date);
  const min = "2020-02-01";
  const max = "2020-02-22";
  const prevDisabled = date == min;
  const nextDisabled = date == max;
  const controls = document.getElementById("controls");
  const prevButton = getDateButton("Prev", prevDisabled, prevMonth);
  const dateHeader = document.createElement("span");
  dateHeader.innerHTML = `${dateToStr(date)} - ${dateToStr(end_date)}`;
  dateHeader.className = "dateHeader";
  const nextButton = getDateButton("Next", nextDisabled, nextMonth);
  controls.innerHTML = "";
  controls.appendChild(prevButton);
  controls.appendChild(dateHeader);
  controls.appendChild(nextButton);
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

function keyPressed() {
  if (keyCode === LEFT_ARROW) {
    month -= 1;
  } else if (keyCode === RIGHT_ARROW) {
    month += 1;
  }
  drawMaster();
}

function drawTSNE() {
  document.getElementById("defaultCanvas0").hidden = false;
  background("white");
  fill("black");
  xMax = d3.max(tsne.map((t) => t.x));
  xMin = d3.min(tsne.map((t) => t.x));
  yMax = d3.max(tsne.map((t) => t.y));
  yMin = d3.min(tsne.map((t) => t.y));
  let largest = d3.max([Math.abs(xMin), xMax, Math.abs(yMin), yMax]);
  for (const pt of tsne) {
    let x = (pt.x + largest) / largest / 2;
    let y = (pt.y + largest) / largest / 2;
    circle(x * SIZE, y * SIZE, 2);
  }
}

function reset() {
  document.getElementById("defaultCanvas0").hidden = true;
  document.getElementById("regionsSVGContainer").hidden = true;
}

function mouseClicked() {}

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
  /*
  let t = r.temperature;
  //* 0.2 + (1 - r.elevation) * 0.8;
  let m = r.moisture;
  let mIndex = Math.floor(m * 2);
  let mFrac = m * 2 - mIndex;
  let tIndex = Math.floor(t * 2);
  let tFrac = t * 2 - tIndex;
  let c0 = getTerrainColor(terrainGrid[mIndex][tIndex]);
  let c1 = getTerrainColor(terrainGrid[mIndex][tIndex + 1]);
  let c2 = getTerrainColor(terrainGrid[mIndex + 1][tIndex]);
  let c3 = getTerrainColor(terrainGrid[mIndex + 1][tIndex + 1]);
  let c01 = d3.interpolateRgb(c0, c1)(tFrac);
  let c23 = d3.interpolateRgb(c2, c3)(tFrac);
  return d3.interpolateRgb(c01, c23)(mFrac);
  */
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
      let vBox = `${xExt[0]} ${yExt[0]} ${xExt[1] - xExt[0]} ${
        yExt[1] - yExt[0]
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
            return (color = getTerrainColorInterpolate(r));
          });
      })
      .on("click", (_, r) => drawCard(r.topic));
    regions.exit().remove();

    let getFontSize = (t) => map(Math.sqrt(t.size), 0, 100, 0.005, 0.04);
    let labels = svg
      .select(".text-primary")
      .selectAll("text")
      .data(Object.values(topics));
    labels
      .enter()
      .append("text")
      .html((d) => d.region_name.toUpperCase())
      .style("fill", "#444")
      .style("text-anchor", "middle")
      .attr("x", (d) => d.x)
      .attr("y", (d) => d.y)
      .attr("filter", "url(#dropshadow)")
      .attr("font-size", (d) => `${getFontSize(d)}pt`);
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
        };
      });
      console.log(Object.keys(vertexData).length, "vertexes");
      console.log(regionData.length, "triangles");
      let svg = renderRegionsSVG(regionData, vertexData);
      console.log("done constructing");
      svgPanZoom("#map_main", {
        controlIconsEnabled: true,
        minZoom: 0.5,
        maxZoom: 10,
      });
    });
  });
}

function draw() {}

let handleMouseOver = function (ev, d) {
  d3.select(this).transition().duration(0.1).attr("r", getSize(1.1));
  drawCard(d.id, d);
};

let drawCard = function (topic_id, r) {
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
        `<p class='word' style='font-size:${Math.sqrt(word_freq[1]) + 6}pt'>${
          word_freq[0]
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
        `<p class='word' style='font-size:${Math.sqrt(word_freq[1]) + 6}pt'>${
          word_freq[0]
        }</p>`
    )
    .join("");

  document.getElementById(
    "wordContainer"
  ).innerHTML += `<p>Size: ${topics[topic_id].size}</p>`;
  document.getElementById(
    "wordContainer"
  ).innerHTML += `<p>Temperature/Mean Subjectivity: ${r.temperature}</p>`;

  document.getElementById(
    "wordContainer"
  ).innerHTML += `<p>Moisture/Media Diversity: ${r.moisture}</p><p>Terrain: </p>`;
  let medianodes = "";
  for (mname in topicMediaNames[topic_id]) {
    if (topicMediaNames[topic_id][mname] > 2) {
      medianodes += `<p class='sm'>${mname}: ${topicMediaNames[topic_id][mname]}`;
    }
  }
  document.getElementById("wordContainer").innerHTML +=
    "<h3>Most Relevant Words</h3>";
  document.getElementById("wordContainer").innerHTML += relevant_words;
  document.getElementById("wordContainer").innerHTML +=
    "<h3>Most Common Words</h3>";
  document.getElementById("wordContainer").innerHTML += common_words;
};

let renderArticles = function (articles) {
  let articleList = document.getElementById("articleList");
  let articleNode = document.createElement("div");
  for (article of articles) {
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

/*

function drawLDAVis() {
  let width = 2;
  let height = 2;
  let topics = [];

  let vis = function (topic_points) {
    // array of id, x, y, and eventually sizes
    let gcolor = (function () {
      const scale = d3.scaleOrdinal(d3.schemeCategory10);
      return (d) => scale(parseInt(d.id));
    })();
    const svg = d3
      .create("svg")
      .attr("viewBox", [-width / 2, -height / 2, width, height]);

    const nodes = svg
      .append("g")
      .selectAll("circle")
      .data(topic_points)
      .join("g");

    nodes
      .append("circle")
      .attr("cx", (d) => d.x)
      .attr("cy", (d) => d.y)
      .attr("r", (d) => Math.sqrt(d.total) * 0.001)
      .attr("fill", gcolor)
      .on("mouseover", handleMouseOver)
      .on("mouseout", handleMouseOut);
    nodes
      .append("text")
      .style("font-size", (d) => Math.sqrt(d.total) * 0.001)
      .attr("x", (d) => d.x)
      .attr("y", (d) => d.y)
      .html((d) => d.id);
    return svg.node();
  };

  let tsvname = getName(date, interval, "mds.tsv");
  d3.tsv(tsvname).then((data) => 
    distance_data = data.map((d) => {
      return {
        x: parseFloat(d[0]),
        y: parseFloat(d[1]),
        id: parseInt(d[""]),
      };
    });
    console.log("loading topics");
    d3.json(
      `topics_${TOPIC}_2020-${monthStr}-01_2020-${monthStr2}-01.json`
    ).then((data) => {
      topics = data;
      for (k in distance_data) {
        distance_data[k]["total"] = topics[k]["total"];
        delete topics[k]["total"];
      }
      let node = vis(distance_data);
      document.getElementById("graphContainer").innerHTML = `<h2>${month}</h2>`;
      document.getElementById("graphContainer").appendChild(node);
    });
  });
}

function drawForceGraph() {
  let width = 1024;
  let height = 1024;
  let gcolor = (function () {
    const scale = d3.scaleOrdinal(d3.schemeCategory10);
    return (d) => scale(parseInt(d.id));
  })();
  let graph = function (data) {
    const links = data.links.map((d) => {
      return {
        source: parseInt(d.source),
        target: parseInt(d.target),
        value: parseFloat(d.value),
      };
    });
    const nodes = data.nodes.map((d) => Object.create(d));

    const simulation = d3
      .forceSimulation(nodes)
      .force(
        "link",
        d3.forceLink(links).id((d) => d.id)
      )
      //.force("collision", d => Math.sqrt(topicSizes[int(d.id)] * 0.3))
      .force(
        "charge",
        d3
          .forceManyBody()
          .strength((d) => -Math.sqrt(topicSizes[int(d.id)]) * 1.2)
      )
      .force("center", d3.forceCenter(width / 2, height / 2).strength(1));

    const svg = d3.create("svg").attr("viewBox", [0, 0, width, height]);

    const link = svg
      .append("g")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.6)
      .selectAll("line")
      .data(links);

    const linklines = link
      .join("line")
      .attr("stroke-width", (d) => Math.sqrt(d.value));
    const linklabels = link
      .join("text")
      .attr("class", "linklabel")
      .html((d) => d.value.toFixed(2));

    const node = svg
      .append("g")
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.5)
      .selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", (d) => Math.sqrt(topicSizes[int(d.id)] * 100))
      .attr("fill", gcolor)
      .on("mouseover", handleMouseOver)
      .on("mouseout", handleMouseOut);

    node.append("title").text((d) => d.id);

    simulation.on("tick", () => {
      linklines
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);
      linklabels
        .attr("x", (d) => (d.source.x + d.target.x) / 2)
        .attr("y", (d) => (d.source.y + d.target.y) / 2);

      node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
    });

    //invalidation.then(() => simulation.stop());

    return svg.node();
  };
  let jsonName = getName(date, interval, "topic_adjacency.json");
  console.log(jsonName);
  d3.json(jsonName).then((data) => {
    document.getElementById("graphContainer").appendChild(graph(data));
  });
}

function drawRegions() {
  document.getElementById("defaultCanvas0").hidden = false;
  background("white");
  for (r of region_data) {
    noStroke();
    fill(230, 250, 255);
    if (r.elevation > 0) {
      fill(200 - r.elevation * 2);
    }
    beginShape();
    for (c of r.coordinates) {
      if (c != -1) {
        vertex(vertex_data[c].x * 1024, vertex_data[c].y * 1024);
      }
    }
    endShape(CLOSE);
  }
}
*/
