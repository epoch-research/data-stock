import {
  run,
  squiggleExpression,
  runForeign,
  Distribution,
  runPartial,
} from "@quri/squiggle-lang";
import { tagged } from "@quri/squiggle-lang/dist/src/js/types";
import { record } from "@quri/squiggle-lang/dist/src/rescript/ReducerInterface/ReducerInterface_ExternalExpressionValue.gen";
import {
  lambdaDeclaration,
  lambdaValue,
} from "@quri/squiggle-lang/dist/src/rescript/TypescriptInterface.gen";
import { readFile, writeFile } from "fs/promises";
import { accessSync, constants } from "fs";

type TaggedRecord = tagged<"record", record>;
type TaggedFunction = tagged<"lambda", lambdaValue>;

const PRECISION = 1000;
const QPREC = 50;
const YEAR_START = 2022;
const YEAR_END = 2030;
const YEAR_INCREMENTS = 0.1;

function checkFileExistsSync(filepath) {
  let flag = true;
  try {
    accessSync(filepath, constants.F_OK);
  } catch (e) {
    flag = false;
  }
  return flag;
}

function evaluateAt(
  f: tagged<"lambda", lambdaValue>,
  y: number,
  n: number,
  transform: (number) => number = (x) => x
) {
  let val = f.value as lambdaValue;
  let runres = runForeign(val, [y], {
    sampleCount: 1000,
    xyPointLength: 100,
  });
  let dist = (runres.value as squiggleExpression).value as unknown as any;
  //if (y == 2080) {
  console.log(runres);
  console.log(dist);
  //console.log(dist.t.value.value);
  //}
  let { xs, ys } = dist.t.value.value.xyShape;
  return {
    xs: xs.map((x) => x.toPrecision(7)),
    ys: ys.map((y) => y.toPrecision(7)),
  };
}

async function loadComputeProjection(bindings, domain: "lang" | "vision") {
  let compute = [];
  let data = await readFile("data/compute_projection.csv", "utf8");
  data
    .toString()
    .split(/\r?\n/)
    .forEach((v, i) => {
      let line = v.split(",");
      let year = line[2];
      let comp = line[1];
      if (!comp) console.log(line);
      compute[Number.parseInt(year) - 2022] = [
        ...(compute[Number.parseInt(year) - 2022] || []),
        comp.substring(0, 6),
      ];
    });
  compute.forEach((v, i) => {
    compute[i] = v.slice(0, 1000);
  });
  let newBindings = runPartial(
    `
    compute_table = ${JSON.stringify(compute).replace(/"/g, "")}
    compute_dists = map(List.upTo(2022,2030), {|y| PointSet.fromDist(SampleSet.fromList(compute_table[y-2022]))})
    compute(t) = compute_dists[min([t-2022,8])] 
    ${
      domain === "lang"
        ? `
      // language
      largest_dataset(t) = (compute(t) - log(6*20)/Math.ln10)*(1/2) + log(20)/Math.ln10
      `
        : `
      // vision
      largest_dataset(t) = utils.psSum((compute(t) - log(6)/Math.ln10)*(0.5), -(2 to 3),50)
    `
    }
  `,
    bindings,
    {
      sampleCount: PRECISION,
      xyPointLength: 100, //PRECISION,
    }
  ).value as record;
  console.log("Loaded compute projection");

  let res = run("largest_dataset", newBindings, {
    sampleCount: PRECISION,
    xyPointLength: 100, //PRECISION,
  });
  console.log(res);
  let fun = res.value as unknown as TaggedFunction;
  let cache = [];
  console.log("Creating cache...");
  for (let y = YEAR_START; y <= YEAR_END; y += YEAR_INCREMENTS) {
    console.log((100 * (y - YEAR_START)) / (YEAR_END - YEAR_START), "%");
    cache.push(evaluateAt(fun, y, PRECISION));
  }
  await writeFile(
    `cache/datasets/${domain}_comp_${PRECISION}_${PRECISION}_${YEAR_START}_${YEAR_END}_${YEAR_INCREMENTS}.pointset`,
    JSON.stringify(cache),
    "utf8"
  );

  return newBindings;
}

async function plotIntersection(
  bindings,
  domain,
  type,
  ystart = YEAR_START,
  yend = YEAR_END,
  prec = PRECISION
) {
  let code = await readFile("src/intersection.squiggle", "utf8");
  console.log("Read intersection code");
  //console.log(bindings);
  let params = runPartial(
    `
    start = ${ystart}
    end = ${yend}
    prec = ${prec}
    `,
    bindings
  ).value as record;
  console.log("Loaded intersection params:" /*, params*/);
  let res = run(code, params, {
    sampleCount: PRECISION,
    xyPointLength: PRECISION,
  });
  console.log(res);
  let samples = (res.value as tagged<"distribution", Distribution>).value.t;
  console.log("Computed intersection");
  writeFile(
    `results/out_intersect_${domain}_${type}_${ystart}_${yend}_${prec}.json`,
    JSON.stringify(samples),
    "utf8"
  );
}

async function loadBindings(
  file: string,
  bindings: record = null,
  samplePrec: number = PRECISION,
  pointPrec: number = PRECISION
): Promise<record> {
  let newBindings: record;
  console.log(`Loading ${file}`);
  if (
    false &&
    checkFileExistsSync(`cache/${file}_${samplePrec}_${pointPrec}.cache`)
  ) {
    newBindings = await JSON.parse(
      await readFile(`cache/${file}_${samplePrec}_${pointPrec}.cache`, "utf8")
    );
  } else {
    console.log("Bindings do not exist");
    let data = await readFile("src/" + file + ".squiggle", "utf8");
    let res = runPartial(data, bindings, {
      sampleCount: samplePrec,
      xyPointLength: pointPrec,
    });
    console.log("Done");
    if (res.tag == "Error") {
      console.log(`Error loading ${file}: ${res.value}`);
    }
    newBindings = res.value as record;
    /*writeFile(
      `cache/${file}_${samplePrec}_${pointPrec}.cache`,
      JSON.stringify(newBindings),
      "utf8"
    );*/
    if (file === "utils") {
      newBindings = { utils: newBindings.utils };
    }
  }
  console.log(`Loaded ${file}`);
  return newBindings;
}

async function loadCachedFunction(
  file: string,
  fn_name: string,
  bindings: record = null,
  samplePrec: number = PRECISION,
  pointPrec: number = PRECISION
): Promise<record> {
  let newBindings: record;
  let cache: { xs: string[]; ys: string[] }[];
  console.log(`Loading ${file}`);
  if (
    checkFileExistsSync(
      `cache/${file}_${samplePrec}_${pointPrec}_${YEAR_START}_${YEAR_END}_${YEAR_INCREMENTS}.pointset`
    )
  ) {
    cache = await JSON.parse(
      await readFile(
        `cache/${file}_${samplePrec}_${pointPrec}_${YEAR_START}_${YEAR_END}_${YEAR_INCREMENTS}.pointset`,
        "utf8"
      )
    );
  } else {
    console.log("Cached function does not exist");
    let data = await readFile("src/" + file + ".squiggle", "utf8");
    let res = run(data, bindings, {
      sampleCount: samplePrec,
      xyPointLength: pointPrec,
    });
    if (res.tag !== "Ok") {
      throw console.error(
        `Error loading ${file}: ${JSON.stringify(res.value)}`
      );
    }
    let fun = ((res.value as unknown as TaggedRecord).value as record)[
      fn_name
    ] as unknown as TaggedFunction;
    console.log("Creating cache...");
    console.log(fun);
    cache = [];
    for (let y = YEAR_START; y <= YEAR_END; y += YEAR_INCREMENTS) {
      console.log((100 * (y - YEAR_START)) / (YEAR_END - YEAR_START), "%");
      cache.push(evaluateAt(fun, y, samplePrec));
    }
    await writeFile(
      `cache/${file}_${samplePrec}_${pointPrec}_${YEAR_START}_${YEAR_END}_${YEAR_INCREMENTS}.pointset`,
      JSON.stringify(cache),
      "utf8"
    );
  }
  //console.log(JSON.stringify(cache).substring(0, 100));
  let dta = cache.map((t) =>
    t.xs.map((_, i) => {
      return { x: t.xs[i], y: t.ys[i] };
    })
  );
  //console.log(`${JSON.stringify(dta).replace(/"|\+/g, "").substring(0, 100)}`);
  newBindings = runPartial(
    `${fn_name}(t) = PointSet.makeContinuous(${JSON.stringify(dta).replace(
      /"|\+/g,
      ""
    )}[(t-${YEAR_START})/${YEAR_INCREMENTS}])
    `,
    bindings
  ).value as record;
  //console.log(newBindings);
  //console.log(run(`${fn_name}(2022)`, newBindings));
  console.log(`Loaded ${file}`);
  return newBindings;
}

type model =
  | "iu"
  | "rw"
  | "iw"
  | "wpp"
  | "cc"
  | "hq"
  | "wagg"
  | "ipp"
  | "ex"
  | "iagg";
async function computeDataStockModels(models: model[]) {
  let bindings = await loadBindings("utils");
  if (models.includes("rw")) {
    bindings = await loadCachedFunction(
      "stocks/lang/recorded_speech",
      "stock_recorded_words",
      bindings
    );
  }
  if (models.includes("iu")) {
    bindings = await loadCachedFunction(
      "stocks/lang/internet_users",
      "stock_internet_words",
      bindings
    );
  }
  if (models.includes("iw")) {
    bindings = await loadCachedFunction(
      "stocks/lang/indexed_websites",
      "stock_indexed_web",
      bindings
    );
  }
  if (models.includes("wpp")) {
    bindings = await loadCachedFunction(
      "stocks/lang/popular_platforms",
      "stock_popular_platforms",
      bindings
    );
  }
  if (models.includes("cc")) {
    bindings = await loadCachedFunction(
      "stocks/lang/common_crawl",
      "stock_cc",
      bindings
    );
  }
  if (models.includes("wagg")) {
    bindings = await loadCachedFunction(
      "stocks/lang/aggregation",
      "stock_of_data",
      bindings
    );
  }
  if (models.includes("hq")) {
    bindings = await loadCachedFunction(
      "stocks/lang/high_quality",
      "stock_of_data",
      bindings
    );
  }
  if (models.includes("ipp")) {
    bindings = await loadCachedFunction(
      "stocks/vision/popular_platforms",
      "stock_popular_platforms",
      bindings
    );
  }
  if (models.includes("ex")) {
    bindings = await loadCachedFunction(
      "stocks/vision/external_estimate",
      "stock_external_estimate",
      bindings
    );
  }
  if (models.includes("iagg")) {
    bindings = await loadCachedFunction(
      "stocks/vision/aggregation",
      "stock_of_data",
      bindings
    );
  }
  return bindings;
}

async function computeDatasetGrowthModels(
  models: ("lang" | "vision")[],
  type: "hist" | "comp",
  bindings
) {
  let newBindings;
  //console.log(bindings);
  if (models.includes("lang")) {
    if (type === "hist") {
      newBindings = await loadCachedFunction(
        "datasets/language_datasets",
        "largest_dataset",
        bindings
      );
    } else {
      newBindings = await loadComputeProjection(bindings, "lang");
    }
  }
  if (models.includes("vision")) {
    if (type === "hist") {
      newBindings = await loadCachedFunction(
        "datasets/vision_datasets",
        "largest_dataset",
        bindings
      );
    } else {
      newBindings = await loadComputeProjection(bindings, "vision");
    }
  }
  return newBindings;
}

//runSims(data);
//plotIntersection(data, "hist");

//computeDataStockModels(["ex", "ipp", "iagg"]);
//computeDatasetGrowthModels(["lang"], "comp", null);
computeDataStockModels([
  //"rw",
  //"iw",
  //"iu",
  //"cc",
  //"wpp",
  //"wagg",
  //"ipp",
  //"ex",
  //"iagg",
  "hq",
]).then((bindings) => {
  computeDatasetGrowthModels(["lang"], "hist", bindings).then((bindings) => {
    plotIntersection(bindings, "lang_hq", "hist", 2022, 2030, 20);
  });
});
