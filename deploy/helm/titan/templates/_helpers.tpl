{{/*
Expand the name of the chart.
*/}}
{{- define "titan.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "titan.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "titan.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "titan.labels" -}}
helm.sh/chart: {{ include "titan.chart" . }}
{{ include "titan.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "titan.selectorLabels" -}}
app.kubernetes.io/name: {{ include "titan.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
API labels
*/}}
{{- define "titan.api.labels" -}}
{{ include "titan.labels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
API selector labels
*/}}
{{- define "titan.api.selectorLabels" -}}
{{ include "titan.selectorLabels" . }}
app.kubernetes.io/component: api
{{- end }}

{{/*
Agent labels
*/}}
{{- define "titan.agent.labels" -}}
{{ include "titan.labels" . }}
app.kubernetes.io/component: agent
{{- end }}

{{/*
Agent selector labels
*/}}
{{- define "titan.agent.selectorLabels" -}}
{{ include "titan.selectorLabels" . }}
app.kubernetes.io/component: agent
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "titan.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "titan.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Redis host
*/}}
{{- define "titan.redis.host" -}}
{{- if .Values.redis.enabled }}
{{- printf "%s-redis-master" .Release.Name }}
{{- else }}
{{- .Values.hiveMind.redis.host }}
{{- end }}
{{- end }}

{{/*
ChromaDB host
*/}}
{{- define "titan.chromadb.host" -}}
{{- if .Values.chromadb.enabled }}
{{- printf "%s-chromadb" (include "titan.fullname" .) }}
{{- else }}
{{- .Values.hiveMind.chromadb.host }}
{{- end }}
{{- end }}
