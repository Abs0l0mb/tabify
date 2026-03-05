'use strict';

import { 
    Block, 
    Div, 
    Tools, 
    Api,
    Button,
    DataTableHeadCell,
    DataTableHeadCellSearchData,
    DataTableHeadCellSortData,
    DataTableRow
} from '@src/classes';

export interface DataTableEndpointData {
    [key: string]: any
}

export interface DataTableConfiguration {
    endpoint: string;
    endpointData?: DataTableEndpointData;
    selectable?: boolean;
    rowOptions?: DataTableRowOption[];
    title?: string;
    theme?: string;
    actions?: DataTableAction[];
}

export interface DataTableRowOption {
    text: string;
    event: string;
}

export interface DataTableAction {
    text: string;
    event: string;
}

export type DataTableManifestColumnType = 'STRING' | 'NUMBER' | 'DATE' | 'BOOLEAN';

export interface DataTableManifestColumn {
    name: string,
    description?: string,
    type: DataTableManifestColumnType
    hidden?: true
}

export interface DataTableManifest {
    columns: DataTableManifestColumn[]
}

export type DataTableOperator = '~=' | '<=' | '>=' | '=' | '<' | '>';

export interface DataTableFilter {
    column: string,
    operator: DataTableOperator,
    value: string | number | Date | boolean | null
}

export type DataTableOrderingMethod = 'asc' | 'desc';

export interface DataTableOrdering {
    method?: DataTableOrderingMethod,
    column: string
}

export interface DataTablePagination {
    lastId?: number,
    lastOrderingColumnValue?: any,
    quantity?: number
}

export interface DataTableRequest {
    filters?: DataTableFilter[],
    ordering?: DataTableOrdering,
    pagination?: DataTablePagination
}

export class DataTable extends Div {

    private title: Div;
    private table: Block;
    private head: Block;
    private body: Block;
    private rows: DataTableRow[] = [];
    private headCells: DataTableHeadCell[] = [];
    private actionsContainer: Div;

    private loaderRow: Block;
    private noDataRow: Block;

    private lastScrollTop: number = 0;
    private lastScrollLeft: number = 0;
    private lastScrollLeftRatio: number = 0;

    private fetching: boolean = false;

    private manifest: DataTableManifest;
    private manifestColumns: Map<string, DataTableManifestColumn>;
    private fetcherFiltersByColumns: Map<string, DataTableFilter[]> = new Map();
    private fetcherOrdering: DataTableOrdering | null = null;
    private fetcherPagination: DataTablePagination | null = null;

    static readonly NUMBER = 'NUMBER';
    static readonly STRING = 'STRING';
    static readonly DATE = 'DATE';
    static readonly BOOLEAN = 'BOOLEAN';

    static readonly SORT_DISABLED = 'SORT_DISABLED';
    static readonly SORT_ASCENDING = 'ASC';
    static readonly SORT_DESCENDING = 'DESC';
    
    constructor(public configuration: DataTableConfiguration, parent?: Block) {

        super('data-table', parent);

        this.draw();
    }

    /*
    **
    **
    */
    public async draw() : Promise<void> {

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

        this.table = new Block('table', {
            cellpadding: 0,
            cellspacing: 0
        }, this);

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

        //===============
        //SCROLL LISTENER
        //===============

        this.element.parentNode.addEventListener('scroll', this.onScroll.bind(this));

        //====================
        //DOWNLOADING MANIFEST
        //====================

        this.manifest = await Api.get(`${this.configuration.endpoint}/manifest`, this.configuration.endpointData);

        this.manifestColumns = this.manifest.columns.reduce((acc, item) => {
            acc.set(item.name, item);
            return acc;
        }, new Map());

        //========
        //HEAD ROW
        //========

        this.head = new Block('tr', {
            class: 'head-row'
        }, this.table);
        
        if (this.configuration.rowOptions)
            new Block('td', {
                class: 'options-head-cell'
            }, this.head);

        for (const column of this.manifest.columns) {

            if (column.hidden)
                continue;

            let headCell = new DataTableHeadCell(column.name, this.head);

            headCell.on('search-data', (data: DataTableHeadCellSearchData) => {
                this.onHeadCellSearchData(headCell, data);
            });

            headCell.on('sort-data', (data: DataTableHeadCellSortData) => {
                this.onHeadCellSortData(headCell, data);
            });

            headCell.on('resize', this.restoreScrollLeft.bind(this));

            this.headCells.push(headCell);
        }

        await Tools.sleep(25);

        this.head.setData('displayed', 1);
        
        //====
        //BODY
        //====

        this.body = new Block('tbody', {}, this.table);

        //===========
        //FIRST FETCH
        //===========

        this.setData('populated', 1);

        this.fetch(true);
    }

    /*
    **
    **
    */
    public async fetch(clearRows: boolean = false) : Promise<void> {

        if (this.fetching)
            return;

        this.fetching = true;

        //========================
        //BUILDING FETCHER REQUEST
        //========================

        const quantity = Math.floor(window.innerHeight * (clearRows ? 1.25 : 0.5) / DataTableRow.HEIGHT);

        let filters: DataTableFilter[] = [];

        for (const [column, columnFilters] of this.fetcherFiltersByColumns)
            filters = filters.concat(columnFilters)

        if (clearRows)
            this.fetcherPagination = {};

        this.fetcherPagination!.quantity = quantity;

        const request: DataTableRequest = {
            filters: filters,
            ordering: this.fetcherOrdering ? this.fetcherOrdering : undefined,
            pagination: this.fetcherPagination ? this.fetcherPagination : undefined
        };

        //==========
        //REQUESTING
        //==========

        const rowsData = await Api.post(this.configuration.endpoint, {
            request: request,
            data: this.configuration.endpointData
        });

        //===================
        //HANDLING PAGINATION
        //===================

        if (rowsData.length > 0) {

            const lastRow = rowsData[rowsData.length - 1];

            this.fetcherPagination!.lastId = lastRow.id;

            if (this.fetcherOrdering && this.fetcherOrdering.column)
                this.fetcherPagination!.lastOrderingColumnValue = lastRow[this.fetcherOrdering.column];
        }

        //==========
        //POPULATING
        //==========

        if (clearRows) {

            const newBody = new Block('tbody');

            if ((!rowsData || rowsData.length === 0))
                this.addNoDataRow(newBody);
            
            else {

                for (const data of rowsData) {
                    this.rows.push(new DataTableRow(data, this, newBody));

                    //debug select
                    //if (this.rows.length === 1)
                    //    this.rows[0].select();
                }
            }

            this.element.scrollTop = 0;
            this.body.element.parentNode.replaceChild(newBody.element, this.body.element);

            this.body = newBody;
        }
        else {

            for (const data of rowsData)
                this.rows.push(new DataTableRow(data, this, this.body));
        }

        this.fetching = false;
    }

    /*
    **
    **
    */
    private addNoDataRow(parent: Block) : void {

        const row = new Block('tr', 'no-data-row', parent);
        new Block('td', {
            colspan: 999
        }, row).write('No data to display');
    }

    /*
    **
    **
    */
    private getFiltersFromText(column: string, source: string, availableOperators: DataTableOperator[]) : DataTableFilter[] {

        const filters: DataTableFilter[] = [];

        const parts = source.trim().split('&');

        let disabledChars = availableOperators.join('').split('');

        partsLoop:
        for (let part of parts) {

            part = part.trim();

            if (part === "")
                continue partsLoop;
            
            for (const operator of availableOperators) {

                const potentialOperator = part.slice(0, operator.length);
                const potentialValue = part.slice(operator.length).trim();

                if (potentialOperator === operator && potentialValue.length > 0 && !disabledChars.includes(potentialValue)) {

                    filters.push({
                        column: column,
                        operator: operator,
                        value: potentialValue
                    });

                    continue partsLoop;
                }
            }
        }

        if (source.trim() !== '' && filters.length === 0 && availableOperators.includes('~=')) {

            filters.push({
                column: column,
                operator: '~=',
                value: source.trim()
            });
        }

        return filters;
    }

    /*
    **
    **
    */
    private onHeadCellSearchData(headCell: DataTableHeadCell, data: DataTableHeadCellSearchData) : void {
        
        let filters: DataTableFilter[] = [];

        switch (this.manifestColumns.get(headCell.getColumn())?.type) {
            
            case DataTable.STRING: {

                filters = this.getFiltersFromText(headCell.getColumn(), data.text, ['~=', '=']);
                break;
            }

            case DataTable.NUMBER: 
            case DataTable.DATE: {

                filters = this.getFiltersFromText(headCell.getColumn(), data.text, ['<=', '>=', '=', '<', '>']);
                break;
            }

            case DataTable.BOOLEAN: {

                filters = this.getFiltersFromText(headCell.getColumn(), data.text, ['=']);
                break;
            }
        }

        this.fetcherFiltersByColumns.set(headCell.getColumn(), filters);

        this.fetch(true);
    }

    /*
    **
    **
    */
    private onHeadCellSortData(headCell: DataTableHeadCell, data: DataTableHeadCellSortData) : void {

        for (let cell of this.headCells) {
            if (cell.uid !== headCell.uid)
                cell.resetSort();
        }

        if ([DataTable.SORT_ASCENDING, DataTable.SORT_DESCENDING].includes(data.method)) {

            this.fetcherOrdering = {
                column: headCell.getColumn(),
                method: data.method as DataTableOrderingMethod
            }
        }
        else
            this.fetcherOrdering = null;
        
        this.fetch(true);
    }

    /*
    **
    **
    */
    private onScroll(event: UIEvent) : void {

        if (this.element.parentNode.scrollTop !== this.lastScrollTop)
            this.onScrollY(event);
        
        if (this.element.parentNode.scrollLeft !== this.lastScrollLeft)
            this.onScrollX(event);
        
        this.lastScrollTop = this.element.parentNode.scrollTop;
        this.lastScrollLeft = this.element.parentNode.scrollLeft;
    }

    /*
    **
    **
    */
    private onScrollY(event: UIEvent) : void {

        const scrollBottom = this.element.parentNode.scrollHeight 
         - this.element.parentNode.clientHeight
         - this.element.parentNode.scrollTop;

         if (scrollBottom < this.element.parentNode.clientHeight / 3)
            this.fetch();
    }

    /*
    **
    **
    */
    private onScrollX(event: UIEvent) : void {

        this.lastScrollLeftRatio = this.element.parentNode.scrollLeft * 100 / (this.element.parentNode.scrollWidth - this.element.parentNode.clientWidth);
    }

    /*
    **
    **
    */
    private restoreScrollLeft() : void {

        if (!this.element)
            return;

        this.element.scrollLeft = (this.element.scrollWidth - this.element.clientWidth) * this.lastScrollLeftRatio / 100;
    }

    /*
    **
    **
    */
    public getManifest() : DataTableManifest | null {

        return this.manifest ? this.manifest : null;
    }

    /*
    **
    **
    */
    public getManifestColumns() : Map<string, DataTableManifestColumn> {

        return this.manifestColumns;
    }

    /*
    **
    **
    */
    public getManifestColumn(column: string) : DataTableManifestColumn | null {

        const manifestColumn = this.manifestColumns.get(column);

        return manifestColumn ? manifestColumn : null;
    }

    /*
    **
    **
    */
    public getRows() : DataTableRow[] {

        return this.rows;
    }
}