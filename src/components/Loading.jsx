import React from "react";
import SyncIcon from "@mui/icons-material/Sync";
import "../stylesheet/loading.css";

export const Loading = () => (
  <p style={{ color: "orange" }}>
    <SyncIcon className={"loading-icon"} /> Loading ...
  </p>
);
