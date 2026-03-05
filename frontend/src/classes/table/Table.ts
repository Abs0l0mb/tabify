'use strict';

import { 
    Block, 
    Div, 
    Tools, 
    Api,
    TableHeadCell, TableHeadCellSearchData, TableHeadCellSortData,
    TableRow,
    Button,
    SimpleSelect
} from '@src/classes';

import structuredClone from '@ungap/structured-clone';

export interface TableRowOption {
    text: string;
    event: string;
}

export interface TableAction {
    text: string;
    event: string;
}

export interface TableConfiguration {
    selectable?: boolean;
    rowOptions?: TableRowOption[];
    title?: string;
    theme?: string;
    actions?: TableAction[];
    exportableAsXLSX?: boolean;
    filters?: any[];
    filtersValue?: string;
    hiddenColumns?: string[];
    extraData?: any[];
    extraDataColumnPrefix?: string;
}

export interface TableColumnsData {
    [key: string]: any;
}

export interface TableRowData {
    [key: string]: any;
}

export interface TableSearchData {
    [headCellKey: string]: {
        headCell: TableHeadCell,        
        data: TableHeadCellSearchData
    }
}

export interface TableSortData {
    headCell: TableHeadCell;
    data: TableHeadCellSortData;
}

/**
 * Classe `Table`
 * 
 * Composant graphique principal permettant d’afficher un tableau dynamique avec des
 * fonctionnalités avancées :
 * - affichage progressif des lignes,
 * - tri et recherche par colonne,
 * - filtres,
 * - export XLSX,
 * - actions sur les lignes ou globales.
 * 
 * Elle hérite de la classe `Div` et s’appuie sur les composants `Block`, `Button`,
 * `TableRow`, `TableHeadCell` et `SimpleSelect` pour construire la structure du DOM.
 * 
 * @example
 * ```typescript
 * const table = new Table({
 *   title: "Utilisateurs",
 *   selectable: true,
 *   exportableAsXLSX: true,
 *   actions: [{ text: "Rafraîchir", event: "refresh" }]
 * });
 * 
 * table.on("refresh", () => table.populate(columns, rows));
 * table.appendTo(document.body);
 * ```
 */

export class Table extends Div {

    private title: Div;
    public table: Block;
    public columnsData: TableColumnsData;
    private head: Block;
    private headCells: TableHeadCell[] = [];
    private bodies: Block[] = [];
    public rows: TableRow[] = [];
    private rowsData: TableRowData[] = [];
    private selectedRowsData: TableRowData[] = [];
    private leftRowsData: TableRowData[] = [];
    private actionsContainer: Div;
    private downloadXLSXButton: Button;
    private filtersContainer: Div;
    private scrollCallback: (scrollBottom: number) => void = () => {};
    private lastScrollTop: number = 0;
    private lastScrollLeft: number = 0;
    private lastScrollLeftRatio: number = 0;
    private searchData: TableSearchData = {};
    private sortData: TableSortData;
    private stickyBodies: Block[] = [];

    static readonly NUMBER = 1;
    static readonly STRING = 2;
    static readonly DATE = 3;
    static readonly PERCENT = 4;

    static readonly SORT_DISABLED = 0;
    static readonly SORT_ASCENDING = 1;
    static readonly SORT_DESCENDING = 2;
    
    /**
     * Crée un nouveau composant de tableau.
     * 
     * @param {TableConfiguration} configuration - Configuration du tableau (colonnes, filtres, actions, etc.).
     * @param {Block} [parent] - Élément parent optionnel dans lequel insérer la table.
     */
    constructor(public configuration: TableConfiguration, parent?: Block) {

        super('table', parent);

        //===============
        //THEME ATTRIBUTE
        //===============

        if (this.configuration.theme)
            this.setData('theme', this.configuration.theme);

        //====================
        //SELECTABLE ATTRIBUTE
        //====================

        if (this.configuration.selectable)
            this.setData('selectable', this.configuration.selectable ? 1 : 0);

        //=====
        //TITLE
        //=====

        if (this.configuration.title) {
            this.title = new Div('title', this);
            new Block('span', {}, this.title).write(this.configuration.title);
        }

        //=====
        //TABLE
        //=====

        if (Array.isArray(this.configuration.filters))
            this.filtersContainer = new Div('filters-container', this);

        this.table = new Block('table', {
            cellpadding: 0,
            cellspacing: 0
        }, this);

        //=======
        //FILTERS
        //=======

        if (Array.isArray(this.configuration.filters)) {

            for (let data of this.configuration.filters) {
                
                const filter = new SimpleSelect({
                    label: data.label,
                    items: data.items,
                    value: data.value,
                    class: 'filter'
                }, this.filtersContainer);

                filter.on('value', async (value) => {
                    this.setData('populated', 0);
                    this.displayLoaderRow();
                    await Tools.sleep(250);
                    this.emit(data.event, value);
                });
            }
        }

        //==========
        //LOADER ROW
        //==========

        this.displayLoaderRow();

        //=======
        //ACTIONS
        //=======

        if (this.configuration.actions) {
            
            this.actionsContainer = new Div('actions-container', this);

            this.setData('has-actions', 1);

            for (let action of this.configuration.actions) {

                let button = new Button({
                    label: action.text
                }, this.actionsContainer).onNative('click', () => {
                    this.emit(action.event, button);
                });
            }
        }

        //====================
        //DOWNLOAD XLSX ACTION
        //====================

        if (this.configuration.exportableAsXLSX) {

            if (!this.actionsContainer)
                this.actionsContainer = new Div('actions-container', this);

            this.downloadXLSXButton = new Button({
                label: "Export"
            }, this.actionsContainer).onNative('click', this.downloadXLSX.bind(this));
        }

        //===============
        //SCROLL LISTENER
        //===============

        if (this.element.parentElement)
            this.element.parentElement.addEventListener('scroll', this.onScroll.bind(this));
    }

    /**
     * Remplit le tableau avec les colonnes et les lignes de données.
     * Efface le contenu précédent, puis affiche la nouvelle structure avec en-têtes et lignes.
     *
     * @param {TableColumnsData} columnsData - Définition des colonnes (nom → type).
     * @param {TableRowData[]} rowsData - Liste des lignes à afficher.
     * @returns {Promise<void>}
     */
    public async populate(columnsData: TableColumnsData, rowsData: TableRowData[]) : Promise<void> {

        this.columnsData = {};
        this.headCells = [];
        this.bodies = [];
        this.rows = [];
        this.rowsData = [];
        this.selectedRowsData = [];
        this.leftRowsData = [];
        this.scrollCallback = () => {};
        this.lastScrollTop = 0;
        this.lastScrollLeft = 0;
        this.lastScrollLeftRatio = 0;
        this.searchData = {};

        this.table.empty();

        this.downloadXLSXButton?.setData('displayed', rowsData.length > 0 ? 1 : 0);

        //==========
        //IF NO ROWS
        //==========

        if (!rowsData || rowsData.length === 0) {
            this.displayNoDataRow();
            this.setData('populated', 1);
            return;
        }

        //=======
        //IF ROWS
        //=======

        this.columnsData = columnsData;
        
        this.rowsData = [];
        let id = 1;
        for (let key in rowsData) {
            rowsData[key].__TABLE_ROW_ID__ = id;
            id++;
        }
        this.rowsData = rowsData;

        if (!this.configuration.hiddenColumns)
            this.configuration.hiddenColumns = [];

        this.configuration.hiddenColumns.push('__TABLE_ROW_ID__');

        //========
        //HEAD ROW
        //========

        this.head = new Block('tr', {
            class: 'head-row'
        }, this.table);
        
        if (this.configuration.rowOptions)
            new Block('th', {
                class: 'options-head-cell'
            }, this.head);

        for (let column in columnsData) {
            
            if (this.configuration.hiddenColumns && this.configuration.hiddenColumns.includes(column))
                continue;

            let headCell = new TableHeadCell(column, this.head);

            headCell.on('search-data', (data: TableHeadCellSearchData) => {
                this.onHeadCellSearchData(headCell, data);
            });

            headCell.on('sort-data', (data: TableHeadCellSortData) => {
                this.onHeadCellSortData(headCell, data);
            });

            headCell.on('resize', this.restoreScrollLeft.bind(this));

            this.headCells.push(headCell);
        }

        await Tools.sleep(25);
        this.head.setData('displayed', 1);
        
        //====
        //ROWS
        //====

        this.displayRowsGradually(rowsData);

        //===============
        //POPULATION DONE
        //===============

        this.setData('populated', 1);
    }

    /**
     * Affiche progressivement les lignes du tableau pour de meilleures performances
     * (utile pour les grands volumes de données).
     * 
     * @param {TableRowData[]} rowsData - Lignes à afficher progressivement.
     */
    private async displayRowsGradually(rowsData: TableRowData[]) : Promise<void> {

        for (let body of this.bodies)
            body.delete();

        this.scrollCallback = (scrollBottom: number) => {};
        
        if (this.element.parentElement)
            this.element.parentElement.scrollTop = 0;

        this.rows = [];
        this.leftRowsData = [...rowsData];

        this.displaySomeRows(false);

        this.scrollCallback = (scrollBottom: number) => {
            if (scrollBottom < 225)
                this.displaySomeRows(true);
        }
    }

    /**
     * Affiche un lot de lignes (scrollDisplay = true si déclenché par le scroll).
     *
     * @param {boolean} scrollDisplay - Indique si l’affichage est lié à un scroll.
     */
    private async displaySomeRows(scrollDisplay: boolean) : Promise<void> {

        this.setData('empty', 0);

        const body = new Block('tbody', {}, this.table);
        
        let displayCount: number;

        if (scrollDisplay) {
            body.addClass('scroll-display');
            displayCount = Math.floor(window.innerHeight * 0.5 / TableRow.HEIGHT);
        }
        else
            displayCount = Math.floor(window.innerHeight * 1.25 / TableRow.HEIGHT);
        
        this.bodies.push(body);
        for (let i=0; i<displayCount; i++) {
            
            const rowData = this.leftRowsData.shift();
            
            if (!rowData)
                break;

            const row = new TableRow(rowData, this, body);

            row.on('update-rows-data', this.updateRowsData.bind(this));

            this.rows.push(row);
        }
        
        this.table.setData('display-fix', parseInt(this.table.getData('display-fix')) === 1 ? 0 : 1);

        await Tools.sleep(25);
        
        body.setData('displayed', 1);

        for (let stickyBody of this.stickyBodies)
            this.table.element.appendChild(stickyBody.element);
    }

    /**
     * Ajoute une ligne "sticky" (fixée en haut du tableau).
     *
     * @param {any} rowData - Données de la ligne à ajouter.
     * @returns {Promise<TableRow>} - L’instance de la ligne ajoutée.
     */
    protected async addStickyRow(rowData: any) : Promise<TableRow> {

        const stickyBody = new Block('tbody', 'sticky', this.table);

        this.stickyBodies.push(stickyBody);

        const row = new TableRow(rowData, this, stickyBody);

        row.on('update-rows-data', this.updateRowsData.bind(this));

        this.rows.push(row);

        await Tools.sleep(25);
        
        stickyBody.setData('displayed', 1);

        return row;
    }

    /** Affiche une ligne indiquant qu’aucune donnée n’est disponible. */
    private displayNoDataRow() : void {

        let noDataRow = new Block('tr', 'no-data-row', this.table.empty());
        new Block('td', {}, noDataRow).write('No data to display');

        this.setData('empty', 1);
    }

    /** Affiche une ligne de chargement temporaire. */
    private async displayLoaderRow() : Promise<void> {

        let loaderRow = new Block('tr', 'loader-row', this.table.empty());
        new Block('td', {}, loaderRow);
    }

    /**
     * Gestion des données de recherche envoyées depuis un `TableHeadCell`.
     * Met à jour la structure interne et relance le filtrage.
     *
     * @param {TableHeadCell} headCell - Cellule concernée.
     * @param {TableHeadCellSearchData} data - Données de recherche.
     */
    private onHeadCellSearchData(headCell: TableHeadCell, data: TableHeadCellSearchData) : void {

        this.searchData[headCell.uid] = {
            headCell: headCell,
            data: data
        };

        this.computeSearchAndSort();
    }

    /**
     * Gestion des données de tri envoyées depuis un `TableHeadCell`.
     * Met à jour l’état de tri et applique le tri sur les lignes affichées.
     *
     * @param {TableHeadCell} headCell - Cellule concernée.
     * @param {TableHeadCellSortData} data - Données de tri.
     */
    private onHeadCellSortData(headCell: TableHeadCell, data: TableHeadCellSortData) : void {

        this.sortData = {
            headCell: headCell,
            data: data
        };

        for (let cell of this.headCells) {
            if (cell.uid !== headCell.uid)
                cell.resetSort();
        }

        this.computeSearchAndSort();
    }

    /**
     * Applique la recherche et le tri sur les données du tableau
     * et rafraîchit les lignes affichées.
     */
    private computeSearchAndSort() : void {

        let selectedRowsData: TableRowData[] = [];
        let rowsDataCopy: TableRowData[] = [...this.rowsData];
        
        //=========
        //SEARCHING
        //=========

        if (this.searchData) {
        
            rowsLoop:
            for (let rowData of rowsDataCopy) {

                headCellsLoop:
                for (let searchData of Object.values(this.searchData)) {

                    const {headCell, data} = searchData;

                    if (!headCell)
                        continue headCellsLoop;

                    if (rowData[headCell.column] === null || rowData[headCell.column] === undefined)
                        rowData[headCell.column] = 'null';

                    const columnType = this.columnsData[headCell.column];

                    if (columnType === Table.STRING || columnType === Table.PERCENT) {
                        if (data.text.trim() === '' || rowData[headCell.column]?.toString().toLowerCase().search(new RegExp(data.text.toLowerCase(), 'g')) !== -1)
                            continue headCellsLoop;
                        else
                            continue rowsLoop;
                    }
                    else if (columnType === Table.DATE) {

                        const subject = new Date(rowData[headCell.column])?.toLocaleString('fr');

                        if (data.text.trim() === '' || subject?.toString().toLowerCase().search(new RegExp(data.text.toLowerCase(), 'g')) !== -1)
                            continue headCellsLoop;
                        else
                            continue rowsLoop;
                    }
                    else if (columnType === Table.NUMBER) {
                        if (data.text.trim() === '' || rowData[headCell.column].toString() === data.text)
                            continue headCellsLoop;
                        else
                            continue rowsLoop;
                    }
                    else
                        continue rowsLoop;
                }

                selectedRowsData.push(rowData);
            }
        }
        else 
            selectedRowsData = rowsDataCopy;

        //=======
        //SORTING
        //=======

        if (this.sortData) {
        
            if ([Table.SORT_ASCENDING, Table.SORT_DESCENDING].includes(this.sortData.data.sort))
                selectedRowsData = Table.sort(selectedRowsData, this.sortData.headCell.column, this.sortData.data.sort, this.getColumnType(this.sortData.headCell));
        }

        //==========
        //COMMITTING
        //==========

        this.selectedRowsData = selectedRowsData;

        this.displayRowsGradually(selectedRowsData);

        this.emit('rows-updated', selectedRowsData);
    }

    /**
     * Récupère le type de données d’une colonne (STRING, NUMBER, DATE, etc.).
     *
     * @param {TableHeadCell} headCell - Cellule d’en-tête concernée.
     * @returns {number} - Type de colonne.
     */
    private getColumnType(headCell: TableHeadCell) : number {

        for (let column in this.columnsData) {
            if (headCell.column === column)
                return this.columnsData[column];
        }

        return 0;
    }

    /**
     * Exporte le contenu du tableau au format XLSX et déclenche le téléchargement.
     * 
     * @returns {Promise<void>}
     */
    public async downloadXLSX() : Promise<void> {

        const data = structuredClone(this.rowsData);

        for (let rowKey in data) {

            if (this.configuration.extraData) {
                let prefix = this.configuration.extraDataColumnPrefix ? this.configuration.extraDataColumnPrefix + ' ' : '';
                for (let column of Object.keys(this.configuration.extraData))
                    data[rowKey][prefix + column] = this.configuration.extraData[column];
            }

            for (let column in data[rowKey]) {

                if (this.columnsData[column] === Table.DATE)
                    data[rowKey][column] = new Date(data[rowKey][column]).toLocaleString('fr');
            }
        }

        const base64 = await Api.post('/json-to-xlsx-base64', {
            data: btoa(JSON.stringify(data))
        });

        const blob = Tools.base64ToBlob(base64, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet');

        const a = document.createElement("a");

        a.href = URL.createObjectURL(blob);

        a.download = "wft-export.xlsx";
        document.body.appendChild(a);
        a.click(); 
        document.body.removeChild(a);

        this.downloadXLSXButton.unload();
    }

    /**
     * Trie un tableau de lignes selon une colonne donnée et un ordre de tri.
     *
     * @param {TableRowData[]} input - Lignes à trier.
     * @param {string} column - Nom de la colonne.
     * @param {number} sort - Sens du tri (`SORT_ASCENDING` ou `SORT_DESCENDING`).
     * @param {number} columnType - Type de données de la colonne.
     * @returns {TableRowData[]} - Lignes triées.
     */
    static sort(input: TableRowData[], column: string, sort: number, columnType: number) : TableRowData[] {

        switch(columnType) {

            case Table.NUMBER:
                return Table.numberSort(input, column, sort);
                break;

            case Table.STRING:
                return Table.stringSort(input, column, sort);
                break;

            case Table.DATE:
                return Table.dateSort(input, column, sort);
                break;

            case Table.PERCENT:
                return Table.percentSort(input, column, sort);
        }

        return input;
    }
    
    /** Trie des valeurs numériques. */
    static numberSort(input: TableRowData[], column: string, sort: number) : TableRowData[] {

        return input.sort((a, b) => {
            if (sort === Table.SORT_ASCENDING)
                return a[column] - b[column];
            else
                return b[column] - a[column];
        });
    }

    /** Trie des valeurs textuelles (ordre alphabétique). */
    static stringSort(input: TableRowData[], column: string, sort: number) : TableRowData[] {
        
        return input.sort((a, b) => {
            if (!a[column]) a[column] = 'null';
            if (!b[column]) b[column] = 'null';
            if (sort === Table.SORT_ASCENDING)
                return a[column].toString().localeCompare(b[column]);
            else
                return b[column].toString().localeCompare(a[column]);
        });
    }

    /** Trie des dates (du plus ancien au plus récent ou inversement). */
    static dateSort(input: TableRowData[], column: string, sort: number) : TableRowData[] {
        
        return input.sort((a, b) => {
            if (sort === Table.SORT_ASCENDING)
                return new Date(a[column]).getTime() - new Date(b[column]).getTime();
            else
                return new Date(b[column]).getTime() - new Date(a[column]).getTime();
        });
    }

    /** Trie des pourcentages (en tant que nombres flottants). */
    static percentSort(input: TableRowData[], column: string, sort: number) : TableRowData[] {

        return input.sort((a, b) => {
            if (sort === Table.SORT_ASCENDING)
                return parseFloat(a[column]) - parseFloat(b[column]);
            else
                return parseFloat(b[column]) - parseFloat(a[column]);
        });
    }

    /**
     * Gestion du scroll principal du tableau (horizontal et vertical).
     * 
     * @param {UIEvent} event - Événement de défilement.
     */
    private onScroll(event: UIEvent) : void {

        if (!this.element.parentElement)
            return; 

        if (this.element.parentElement.scrollTop !== this.lastScrollTop)
            this.onScrollY(event);
        
        if (this.element.parentElement.scrollLeft !== this.lastScrollLeft)
            this.onScrollX(event);
        
        this.lastScrollTop = this.element.parentElement.scrollTop;
        this.lastScrollLeft = this.element.parentElement.scrollLeft;
    }

    /** Gestion du défilement vertical (déclenche le chargement progressif). */
    private onScrollY(event: UIEvent) : void {

        if (!this.element.parentElement)
            return; 

        const scrollBottom = this.element.parentElement.scrollHeight 
         - this.element.parentElement.clientHeight
         - this.element.parentElement.scrollTop;

        this.scrollCallback(scrollBottom);
    }

    /** Gestion du défilement horizontal (sauvegarde la position pour restauration). */
    private onScrollX(event: UIEvent) : void {

        if (!this.element.parentElement)
            return; 

        this.lastScrollLeftRatio = this.element.parentElement.scrollLeft * 100 / (this.element.parentElement.scrollWidth - this.element.parentElement.clientWidth);
    }

    /** Restaure la position horizontale du scroll après un redimensionnement. */
    private restoreScrollLeft() : void {

        if (!this.element.parentElement)
            return; 

        this.element.parentElement.scrollLeft = (this.element.parentElement.scrollWidth - this.element.parentElement.clientWidth) * this.lastScrollLeftRatio / 100;
    }

    /**
     * Met à jour les données d’une ligne après modification.
     * 
     * @param {TableRowData} updatedRowData - Données mises à jour.
     */
    private updateRowsData(updatedRowData) : void {

        for (let key in this.rowsData) {
            
            if (this.rowsData[key].__TABLE_ROW_ID__ === updatedRowData.__TABLE_ROW_ID__)
                this.rowsData[key] = updatedRowData;
        }
    }
}