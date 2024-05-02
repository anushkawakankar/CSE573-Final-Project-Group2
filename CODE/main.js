movies_path = "../DATA/movies.csv";
users_path = "../DATA/users.csv";
const backend = "http://127.0.0.1:9999";
let users;
let movies;

async function getResultFromBackend(url, callback) {
  const response = await fetch(url);
  const json = await response.json();
  callback(json);
}

document.addEventListener("DOMContentLoaded", function () {
  user_select = document.getElementById("user");
  movie_select = document.getElementById("movie");
  cold_start = document.getElementById("coldstart");
  rmovies = document.getElementById("rmovies");
  rusers = document.getElementById("rusers");
  prating = document.getElementById("prating");
  output = document.getElementById("output");

  rmovies.onclick = () => {
    console.log(user_select.value);
    console.log(movie_select.value);
    console.log(cold_start.checked);
    getResultFromBackend(
      `${backend}/recommend_movies/${user_select.value}`,
      (res) => {
        $("#output").html(`Recommended movies for User ${user_select.value}:`);
        console.log(res);
        for (const ii of res["movies"]) {
          const div = document.createElement("div");
          div.innerHTML = ii;
          output.appendChild(div);
        }
      }
    );
  };

  rusers.onclick = () => {
    console.log(user_select.value);
    console.log(movie_select.value);
    console.log(cold_start.checked);
    getResultFromBackend(
      `${backend}/recommend_users/${movie_select.value}`,
      (res) => {
        $("#output").html(
          `Recommended users for ${movies[movie_select.value]["movie title"]}:`
        );
        console.log(res);
        for (const ii of res["users"]) {
          const div = document.createElement("div");
          div.innerHTML = "User " + ii;
          output.appendChild(div);
        }
      }
    );
  };

  prating.onclick = () => {
    console.log(user_select.value);
    console.log(movie_select.value);
    console.log(cold_start.checked);
    getResultFromBackend(
      `${backend}/predict_rating/${user_select.value}/${movie_select.value}`,
      (res) => {
        $("#output").html(
          `Predicted Rating: ${res["predicted_rating"]} <br> Actual Rating: ${res["actual_rating"]}`
        );
        console.log(res);
        // for (const ii of res["users"]) {
        //   const div = document.createElement("div");
        //   div.innerHTML = ii;
        //   output.appendChild(div);
        // }
      }
    );
  };

  Promise.all([
    d3.csv(users_path, d3.autoType),
    d3.csv(movies_path, d3.autoType),
  ]).then((values) => {
    users = values[0];
    movies = values[1];

    for (const user of users) {
      var opt = document.createElement("option");
      opt.value = user["user id"];
      opt.innerHTML = user["user id"];
      user_select.appendChild(opt);
    }

    for (const movie of movies) {
      var opt = document.createElement("option");
      opt.value = movie["movie id"];
      opt.innerHTML = movie["movie title"];
      movie_select.appendChild(opt);
    }

    // document.getElementById("output").innerHTML = "100";
    getResultFromBackend(`${backend}/recommend_movies/1`, (res) =>
      console.log(res)
    );
    getResultFromBackend(`${backend}/recommend_users/1`, (res) =>
      console.log(res)
    );
    getResultFromBackend(`${backend}/predict_rating/1/1`, (res) =>
      console.log(res)
    );
  });
});
